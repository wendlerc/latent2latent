import fsspec
import random
import tempfile
import os
import logging
import subprocess
import json
import threading
import time
import torch
from torch.utils.data import IterableDataset, DataLoader
import torch.distributed as dist
from torchvision import transforms
from torchvision.io import read_video
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor

load_dotenv("/home/developer/workspace/.env2")

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RandomizedQueue:
    def __init__(self):
        self.items = []
        self.lock = threading.Lock()

    def add(self, item):
        with self.lock:
            idx = random.randint(0, len(self.items))
            self.items.insert(idx, item)

    def pop(self):
        with self.lock:
            if not self.items:
                return None
            idx = random.randint(0, len(self.items) - 1)
            return self.items.pop(idx)

    def size(self):
        with self.lock:
            return len(self.items)

class MP4VideoDataset(IterableDataset):
    """
    PyTorch IterableDataset for streaming MP4 video processing from S3.
    
    Returns video tensors in uint8 format [frames_per_sample, 3, H, W] for memory efficiency.
    Uses background threads to extract and process videos concurrently.
    """
    def __init__(self, 
                 bucket_name='cod-yt-playlist', 
                 prefix='',
                 frames_per_sample=101,
                 target_height=360,
                 target_width=640,
                 crop_duration=10,
                 max_videos_queue=5,
                 max_tensors_queue=50,
                 rank=0, 
                 world_size=1):
        super().__init__()
        
        self.bucket_name = bucket_name
        self.prefix = prefix.strip('/')
        self.frames_per_sample = frames_per_sample
        self.target_height = target_height
        self.target_width = target_width
        self.crop_duration = crop_duration
        self.max_videos_queue = max_videos_queue
        self.max_tensors_queue = max_tensors_queue
        self.rank = rank
        self.world_size = world_size
        
        # Setup S3 filesystem
        self.fs = self._get_fs()
        
        # Initialize video transform - keep as uint8 for memory efficiency
        self.transform = transforms.Compose([
            transforms.Resize((target_height, target_width)),
            # No dtype conversion - keep as uint8
        ])
        
        # Initialize queues
        self.video_queue = RandomizedQueue()  # Queue of video file paths
        self.tensor_queue = RandomizedQueue()  # Queue of processed tensors
        
        # Thread pool for concurrent video extraction
        self.video_extractor_pool = ThreadPoolExecutor(max_workers=max_videos_queue, thread_name_prefix="VideoExtractor")
        self.pending_extractions = 0  # Track pending extraction tasks
        self.extraction_lock = threading.Lock()  # Lock for pending_extractions counter
        
        # Thread pool for concurrent tensor processing
        self.tensor_processor_pool = ThreadPoolExecutor(max_workers=max_videos_queue, thread_name_prefix="TensorProcessor")
        self.pending_tensor_processing = 0  # Track pending tensor processing tasks
        self.tensor_processing_lock = threading.Lock()  # Lock for pending_tensor_processing counter
        
        # Shutdown flag for graceful cleanup
        self.shutdown_flag = threading.Event()
        
        # Get available MP4 keys once at startup
        self.available_keys = self._get_all_keys()
        logger.info(f"Found {len(self.available_keys)} MP4 files in bucket")
        
        # Start background threads
        self.video_thread = threading.Thread(target=self._background_extract_videos, daemon=True)
        self.tensor_thread = threading.Thread(target=self._background_process_videos, daemon=True)
        self.video_thread.start()
        self.tensor_thread.start()
        
        logger.info(f"MP4VideoDataset initialized for rank {rank}/{world_size} with {max_videos_queue} concurrent video extractors and {max_videos_queue} concurrent tensor processors")

    def _get_fs(self):
        kwargs = {
            'key': os.getenv('AWS_ACCESS_KEY_ID'),
            'secret': os.getenv('AWS_SECRET_ACCESS_KEY'),
        }
        endpoint = os.getenv('AWS_ENDPOINT_URL_S3')
        if endpoint:
            kwargs['endpoint_url'] = endpoint
        return fsspec.filesystem('s3', **{k:v for k,v in kwargs.items() if v})

    def _get_all_keys(self):
        """Get all available MP4 keys once at startup"""
        if self.prefix:
            pattern = f"{self.bucket_name}/{self.prefix}/**/*.mp4"
        else:
            pattern = f"{self.bucket_name}/**/*.mp4"

        logger.info(f"🔍 Scanning S3 with pattern: {pattern}")
        keys = self.fs.glob(pattern)

        if not keys and self.prefix:
            pattern2 = f"{self.bucket_name}/{self.prefix}/*.mp4"
            logger.info(f"🔍 Trying non-recursive pattern: {pattern2}")
            keys = self.fs.glob(pattern2)

        if not keys:
            raise RuntimeError(f"No MP4 files found in s3://{self.bucket_name}/{self.prefix}")
        
        return keys

    def _extract_video_crop(self, key):
        logger.info(f"Extracting video crop for {key}")
        """Extract a video crop using ffmpeg directly with S3 presigned URL"""
        try:
            # Generate presigned URL
            presigned_url = self.fs.url(key, expires=3600)
            
            # Get video duration using ffprobe
            probe_cmd = [
                'ffprobe', '-v', 'quiet',
                '-print_format', 'json',
                '-show_format',
                presigned_url
            ]
            
            result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
            if result.returncode != 0:
                logger.warning(f"ffprobe failed for {key}: {result.stderr}")
                return None
                
            info = json.loads(result.stdout)
            duration = float(info.get('format', {}).get('duration', 0))
            
            if duration <= self.crop_duration:
                logger.warning(f"Video too short ({duration:.2f}s) for {self.crop_duration}s crop: {key}")
                return None
            
            # Choose random start time
            start = random.uniform(0, duration - self.crop_duration)
            
            # Extract clip using ffmpeg
            out_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
            out_path = out_file.name
            out_file.close()
            """
            ffmpeg_cmd = [
                'ffmpeg', '-y',
                '-ss', str(start),
                '-i', presigned_url,
                '-t', str(self.crop_duration),
                '-c:v', 'libx264',
                '-c:a', 'aac',
                '-r', '30',  # Force 30 FPS for consistent frame counts
                '-pix_fmt', 'yuv420p',
                '-avoid_negative_ts', 'make_zero',
                out_path
            ]
            """
            ffmpeg_cmd = [
                'ffmpeg', '-y',  # overwrite output
                '-ss', str(start),  # seek to start time
                '-i', presigned_url,  # input from S3
                '-t', str(self.crop_duration),  # duration
                '-c:v', 'libx264',  # video codec
                '-c:a', 'aac',  # audio codec
                '-r', '30',  # Force 30 FPS for consistent frame counts
                '-err_detect', 'ignore_err',
                '-fflags', '+igndts',
                '-movflags', '+frag_keyframe+empty_moov',
                '-avoid_negative_ts', 'make_zero',  # handle timing issues
                out_path
            ]
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=300)
            if result.returncode != 0:
                logger.warning(f"ffmpeg failed for {key}: {result.stderr}")
                try:
                    os.unlink(out_path)
                except:
                    pass
                return None
            
            return out_path

        except Exception as e:
            logger.error(f"Error extracting crop from {key}: {e}")
            return None

    def _on_extraction_complete(self, future):
        """Callback for when a video extraction task completes"""
        try:
            video_path = future.result()
            if video_path:
                self.video_queue.add(video_path)
                logger.debug(f"Added video to queue: {video_path}")
        except Exception as e:
            logger.error(f"Video extraction failed: {e}")
        finally:
            # Decrement pending extractions counter
            with self.extraction_lock:
                self.pending_extractions -= 1

    def _on_tensor_processing_complete(self, future):
        """Callback for when a tensor processing task completes"""
        try:
            tensors, video_path = future.result()
            
            # Clean up video file
            try:
                if video_path:
                    os.unlink(video_path)
            except:
                pass
            
            if tensors is not None:
                for tensor in tensors:
                    if tensor is not None:
                        self.tensor_queue.add(tensor)
                        logger.debug(f"Added tensor to queue, shape: {tensor.shape}")
                        
                        # Check if tensor queue is getting full
                        if self.tensor_queue.size() >= self.max_tensors_queue:
                            break
        except Exception as e:
            logger.error(f"Tensor processing failed: {e}")
        finally:
            # Decrement pending tensor processing counter
            with self.tensor_processing_lock:
                self.pending_tensor_processing -= 1

    def _background_extract_videos(self):
        """Background thread to manage concurrent video extractions"""
        while not self.shutdown_flag.is_set():
            try:
                # Check if we need more videos and can start more extractions
                with self.extraction_lock:
                    current_total = self.video_queue.size() + self.pending_extractions
                
                if current_total < self.max_videos_queue:
                    # Pick a random key
                    key = random.choice(self.available_keys)
                    
                    # Submit extraction task to thread pool
                    with self.extraction_lock:
                        self.pending_extractions += 1
                    
                    future = self.video_extractor_pool.submit(self._extract_video_crop, key)
                    future.add_done_callback(self._on_extraction_complete)
                    
                    logger.debug(f"Submitted extraction task for {key}, pending: {self.pending_extractions}")
                else:
                    time.sleep(1)
            except Exception as e:
                if not self.shutdown_flag.is_set():  # Only log if not shutting down
                    logger.error(f"Error in background video extraction management: {e}")
                time.sleep(5)

    def _background_process_videos(self):
        """Background thread to manage concurrent tensor processing"""
        while not self.shutdown_flag.is_set():
            try:
                # Check if we need more tensor processing and can start more tasks
                with self.tensor_processing_lock:
                    current_processing = self.pending_tensor_processing
                
                if (self.tensor_queue.size() < self.max_tensors_queue and 
                    current_processing < self.max_videos_queue):
                    
                    video_path = self.video_queue.pop()
                    if video_path is None:
                        time.sleep(1)
                        continue
                    
                    # Submit tensor processing task to thread pool
                    with self.tensor_processing_lock:
                        self.pending_tensor_processing += 1
                    
                    future = self.tensor_processor_pool.submit(self._process_video_to_tensor_with_path, video_path)
                    future.add_done_callback(self._on_tensor_processing_complete)
                    
                    logger.debug(f"Submitted tensor processing task for {video_path}, pending: {current_processing + 1}")
                else:
                    time.sleep(1)
            except Exception as e:
                if not self.shutdown_flag.is_set():  # Only log if not shutting down
                    logger.error(f"Error in background tensor processing management: {e}")
                time.sleep(5)

    def _process_video_to_tensor(self, video_path):
        """Convert video file to multiple uint8 tensors with target shape [frames_per_sample, 3, H, W]
        
        Returns tensors in uint8 format (0-255 range) for memory efficiency.
        """
        try:
            # Read video using torchvision
            video_tensor, audio_tensor, info = read_video(video_path, pts_unit='sec')
            
            # video_tensor shape is [T, H, W, C], we need [T, C, H, W]
            video_tensor = video_tensor.permute(0, 3, 1, 2)
            
            logger.info(f"Video tensor shape: {video_tensor.shape}")
            logger.info(f"Video tensor min: {video_tensor.min()}, max: {video_tensor.max()}, dtype: {video_tensor.dtype}")
            
            # Keep as uint8 for memory efficiency - no normalization
            # Apply transforms (resize) - works with uint8
            video_tensor = self.transform(video_tensor)
            
            # Calculate how many complete samples we can extract
            total_frames = video_tensor.shape[0]
            if total_frames < self.frames_per_sample:
                logger.warning(f"Video has only {total_frames} frames, need {self.frames_per_sample}")
                return None
            
            # Calculate number of complete samples (no overlap)
            num_samples = total_frames // self.frames_per_sample
            
            # Add some randomization to reduce temporal correlation
            max_offset = min(30, total_frames - (num_samples * self.frames_per_sample))  # Up to 1 second offset
            offset = random.randint(0, max_offset) if max_offset > 0 else 0
            
            tensors = []
            for i in range(num_samples):
                # Calculate start and end indices for this sample
                start_idx = offset + (i * self.frames_per_sample)
                end_idx = start_idx + self.frames_per_sample
                
                # Make sure we don't exceed video length
                if end_idx > total_frames:
                    break
                
                # Extract the required number of frames
                sampled_tensor = video_tensor[start_idx:end_idx]
                tensors.append(sampled_tensor)
                logger.debug(f"Created tensor {i+1}/{num_samples}, shape: {sampled_tensor.shape}")
            
            logger.info(f"Created {len(tensors)} tensors from video")
            return tensors
            
        except Exception as e:
            logger.error(f"Error processing video {video_path}: {e}")
            return None

    def _process_video_to_tensor_with_path(self, video_path):
        """Wrapper to process video and return both tensors and path for cleanup"""
        tensors = self._process_video_to_tensor(video_path)
        return tensors, video_path

    def __iter__(self):
        """Iterator that yields video tensors"""
        while True:
            tensor = self.tensor_queue.pop()
            if tensor is not None:
                yield tensor
            else:
                time.sleep(0.1)

    def cleanup(self):
        """Clean up resources including thread pools"""
        # Signal background threads to stop
        if hasattr(self, 'shutdown_flag'):
            self.shutdown_flag.set()
            logger.info("Shutdown flag set, waiting for background threads to stop...")
            time.sleep(2)  # Give threads time to notice and exit gracefully
        
        if hasattr(self, 'video_extractor_pool'):
            logger.info("Shutting down video extractor thread pool...")
            self.video_extractor_pool.shutdown(wait=True)
            logger.info("Video extractor thread pool shut down successfully")
        
        if hasattr(self, 'tensor_processor_pool'):
            logger.info("Shutting down tensor processor thread pool...")
            self.tensor_processor_pool.shutdown(wait=True)
            logger.info("Tensor processor thread pool shut down successfully")

    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.cleanup()
        except:
            pass  # Ignore errors during cleanup in destructor

def collate_fn(batch):
    """Collate function to stack video tensors"""
    return torch.stack(batch)

def get_video_loader(batch_size=4, **dataset_kwargs):
    """Create a DataLoader for MP4 videos"""
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    dataset = MP4VideoDataset(rank=rank, world_size=world_size, **dataset_kwargs)
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

if __name__ == "__main__":
    # Test the dataset
    logger.info("🚀 Testing MP4VideoDataset with multi-sample extraction")
    
    loader = get_video_loader(
        batch_size=1,  # Reduced since we get more samples per video
        bucket_name='cod-yt-playlist',
        frames_per_sample=101,
        target_height=360,
        target_width=640,
        crop_duration=10.2,
        max_videos_queue=6,  # Fewer videos needed
        max_tensors_queue=(61*30)//101 # More tensors per video
    )
    
    logger.info("⏱️  Loading 5 batches for timing analysis...")
    
    import os
    os.makedirs('/home/developer/workspace/data/pt', exist_ok=True)
    
    timings = []
    for i in range(20):
        start_time = time.time()
        batch = next(iter(loader))
        batch_time = time.time() - start_time
        timings.append(batch_time)
        logger.info(f"Batch {i+1}/20: {batch_time:.2f}s")
        
        # Save batch as .pt file
        torch.save(batch, f'/home/developer/workspace/data/pt/{i:04d}.pt')
        logger.info(f"Saved batch {i+1}/20 to /home/developer/workspace/data/pt/{i:04d}.pt")
    
    avg_time = sum(timings) / len(timings)
    min_time = min(timings)
    max_time = max(timings)
    
    logger.info(f"✅ Timing analysis complete!")
    logger.info(f"✅ Average batch time: {avg_time:.2f}s")
    logger.info(f"✅ Min batch time: {min_time:.2f}s")
    logger.info(f"✅ Max batch time: {max_time:.2f}s")
    logger.info(f"✅ Batch shape: {batch.shape}")
    logger.info(f"✅ Data type: {batch.dtype}")
    logger.info(f"✅ Tensor stats (uint8) - min: {batch.min()}, max: {batch.max()}, mean: {batch.float().mean():.1f}")
    logger.info(f"✅ Memory usage: ~4x less than float32!")
    logger.info(f"✅ Expected: ~2-3 samples per 10s video = 3x more efficient!")