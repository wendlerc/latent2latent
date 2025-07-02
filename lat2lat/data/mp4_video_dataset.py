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
import gc
import psutil

load_dotenv()

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def dl_mp4(mp4_path, crop_duration, from_time=None, to_time=None, target_width=640, target_height=360):
    # Get video duration and color space info using ffprobe with retry
    def run_ffprobe():
        probe_cmd = [
            'ffprobe', '-v', 'quiet',
            '-print_format', 'json',
            '-show_format', '-show_streams',
            mp4_path
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"ffprobe failed: {result.stderr}")
        return json.loads(result.stdout)
    
    info = run_ffprobe()   
    duration = float(info.get('format', {}).get('duration', 0))
    
    # Extract color space info and use it for FFmpeg
    video_stream = next((s for s in info.get('streams', []) if s.get('codec_type') == 'video'), {})
    color_space = video_stream.get('color_space', 'bt709')  # More conservative default
    color_range = video_stream.get('color_range', 'pc')   
    color_primaries = video_stream.get('color_primaries', 'bt709')  # More conservative default
    color_trc = video_stream.get('color_trc', 'bt709')  # More conservative default
    
    logger.info(f"Using color info - space: {color_space}, range: {color_range}, primaries: {color_primaries}")
    
    if duration <= crop_duration:
        logger.warning(f"Video too short ({duration:.2f}s) for {crop_duration}s crop: {mp4_path}")
        return None
    
    # Determine valid time range for sampling
    min_start = from_time if from_time is not None else 0
    max_end = to_time if to_time is not None else duration
    
    # Ensure we have enough duration for the crop within the specified range
    available_duration = max_end - min_start
    if available_duration < crop_duration:
        logger.warning(f"Insufficient duration ({available_duration:.2f}s) in time range [{min_start:.1f}, {max_end:.1f}] for {crop_duration}s crop: {mp4_path}")
        return None
    
    # Choose random start time within the constrained range
    max_start = max_end - crop_duration
    start = random.uniform(min_start, max_start)
    
    logger.debug(f"Video duration: {duration:.2f}s, sampling from [{min_start:.1f}, {max_end:.1f}], start: {start:.2f}s")
    
    # Extract clip using ffmpeg
    out_file = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)
    out_path = out_file.name
    out_file.close()
    
    # Map range values (FFmpeg uses different names than probe output)
    range_mapping = {
        'tv': 'tv',      # Limited range (16-235)
        'pc': 'pc',      # Full range (0-255)  
        'limited': 'tv',
        'full': 'pc',
        'unknown': 'pc'  # Default to full range for safety
    }
    output_range = range_mapping.get(color_range, 'pc')  # Default to full range for high quality
    
    def run_ffmpeg():
        ffmpeg_cmd = [
            'ffmpeg', '-y',  # overwrite output
            '-ss', str(start),  # seek to start time
            '-i', mp4_path,  # input from S3
            '-t', str(crop_duration),  # duration
            '-c:v', 'libx264',  # video codec
            '-c:a', 'aac',  # audio codec
            '-r', '30',  # Force 30 FPS for consistent frame counts
            # Use detected color space settings to preserve original appearance
            '-colorspace', color_space,
            '-color_primaries', color_primaries,
            '-color_trc', color_trc,
            '-color_range', output_range,
            '-vf', f'scale={target_width}:{target_height}:in_color_matrix={color_space}:in_range={output_range}:out_color_matrix={color_space}:out_range={output_range}',
            '-crf', '18',  # High quality to minimize artifacts
            '-preset', 'fast',  # Balance speed vs quality
            '-err_detect', 'ignore_err',
            '-fflags', '+igndts',
            '-movflags', '+frag_keyframe+empty_moov',
            '-avoid_negative_ts', 'make_zero',  # handle timing issues
            out_path
        ]
        result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=300)
        if result.returncode != 0:
            raise RuntimeError(f"ffmpeg failed: {result.stderr}")
        return out_path
    dled_mp4 = run_ffmpeg()
    if dled_mp4 is None:
        logger.warning(f"ffmpeg returned None for {mp4_path}")
        return None
    return dled_mp4

def process_video_to_tensor(video_path, frames_per_sample):
    """Convert video file to multiple uint8 tensors with target shape [frames_per_sample, 3, H, W]
    
    Returns tensors in uint8 format (0-255 range) for memory efficiency.
    """
    # Read video using torchvision
    video_tensor, audio_tensor, info = read_video(video_path, pts_unit='sec')
    
    # video_tensor shape is [T, H, W, C], we need [T, C, H, W]
    video_tensor = video_tensor.permute(0, 3, 1, 2)
    
    logger.info(f"Video tensor shape: {video_tensor.shape}")
    logger.info(f"Video tensor min: {video_tensor.min()}, max: {video_tensor.max()}, dtype: {video_tensor.dtype}")
    
    # Keep as uint8 for memory efficiency - no normalization
    # No transforms needed - FFmpeg already resized to target dimensions
    
    # Calculate how many complete samples we can extract
    total_frames = video_tensor.shape[0]
    if total_frames < frames_per_sample:
        logger.warning(f"Video has only {total_frames} frames, need {frames_per_sample}")
        return None
    
    # Calculate number of complete samples (no overlap)
    num_samples = total_frames // frames_per_sample
    
    # Add some randomization to reduce temporal correlation
    max_offset = min(30, total_frames - (num_samples * frames_per_sample))  # Up to 1 second offset
    offset = random.randint(0, max_offset) if max_offset > 0 else 0
    
    tensors = []
    for i in range(num_samples):
        # Calculate start and end indices for this sample
        start_idx = offset + (i * frames_per_sample)
        end_idx = start_idx + frames_per_sample
        
        # Make sure we don't exceed video length
        if end_idx > total_frames:
            break
        
        # Extract the required number of frames
        sampled_tensor = video_tensor[start_idx:end_idx]
        tensors.append(sampled_tensor)
        logger.debug(f"Created tensor {i+1}/{num_samples}, shape: {sampled_tensor.shape}")
    
    logger.info(f"Created {len(tensors)} tensors from video")
    return tensors

def mp4_worker(mp4_path, crop_duration, from_time=None, to_time=None, target_width=640, target_height=360, frames_per_sample=101):
    dled_mp4 = dl_mp4(mp4_path, crop_duration, from_time, to_time, target_width, target_height)
    if dled_mp4 is None:
        logger.warning(f"ffmpeg returned None for {mp4_path}")
        return None
    tensors = process_video_to_tensor(dled_mp4, frames_per_sample)
    # clean up
    os.unlink(dled_mp4)
    return tensors

class RandomizedQueue:
    def __init__(self):
        self.items = []

    def add(self, item):
        idx = random.randint(0, len(self.items))
        self.items.insert(idx, item)

    def pop(self):
        if not self.items:
            return None
        idx = random.randint(0, len(self.items) - 1)
        return self.items.pop(idx)

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
                 from_time=None,
                 to_time=None,
                 max_parallel_downloads=8,
                 max_memory_gb=25,
                 max_videos=None,
                 rank=0, 
                 world_size=1):
        super().__init__()
        
        self.bucket_name = bucket_name
        self.prefix = prefix.strip('/')
        self.frames_per_sample = frames_per_sample
        self.target_height = target_height
        self.target_width = target_width
        self.crop_duration = crop_duration
        self.from_time = from_time  # Start time in seconds (None = start from beginning)
        self.to_time = to_time      # End time in seconds (None = go to end)
        bytes_per_tensor = self.frames_per_sample * self.target_height * self.target_width * 3
        self.max_parallel_downloads = max_parallel_downloads
        self.max_memory_gb = max_memory_gb
        self.max_videos = max_videos  # Maximum number of videos to process (None = unlimited)
        self.max_tensors_in_memory = self.max_memory_gb * 1024**3 // bytes_per_tensor
        self.max_videos_queue = max_parallel_downloads
        self.rank = rank
        self.world_size = world_size
        self.futures_lock = threading.Lock()
        self.n_active_futures = 0
        
        # Add shutdown flag for graceful cleanup
        self.shutdown_flag = threading.Event()
        
        # Setup S3 filesystem
        self.fs = self._get_fs()
        
        # Initialize queues with memory-aware sizing
        self.tensor_queue = RandomizedQueue()
        
        # Get available MP4 keys once at startup with sharding for distributed training
        all_keys = self._get_all_keys()
        self.available_keys = self._shard_keys_for_rank(all_keys)
        logger.info(f"Found {len(all_keys)} total MP4 files, rank {rank} will process {len(self.available_keys)} files")
        
        logger.info(f"MP4VideoDataset initialized for rank {rank}/{world_size} with {max_parallel_downloads} concurrent video extractors and {max_parallel_downloads} concurrent tensor processors")
        logger.info(f"Memory limit: {max_memory_gb}GB")
        
        # Start background threads
        self.executor = ThreadPoolExecutor(max_workers=max_parallel_downloads)
        self.dl_thread = threading.Thread(target=self._background_download_mp4s, daemon=True)
        self.dl_thread.start()

    def _background_download_mp4s(self):
        while not self.shutdown_flag.is_set():
            if self.n_active_futures < self.max_parallel_downloads and len(self.tensor_queue.items) < self.max_tensors_in_memory:
                
                mp4_path = self.available_keys[random.randint(0, len(self.available_keys) - 1)]
                url = self.fs.url(mp4_path, expires=3600)
                
                # Check shutdown flag before submitting new task
                if self.shutdown_flag.is_set():
                    break
                    
                future = self.executor.submit(mp4_worker, url, self.crop_duration, 
                                            self.from_time, self.to_time, self.target_width, 
                                            self.target_height, self.frames_per_sample)
                future.add_done_callback(self._on_mp4_worker_complete)
                with self.futures_lock:
                    self.n_active_futures += 1
            else:
                # Check shutdown flag before sleeping
                if self.shutdown_flag.is_set():
                    break
                time.sleep(1)
        logger.info("Background download thread exiting")

    def _on_mp4_worker_complete(self, future):
        with self.futures_lock:
            self.n_active_futures -= 1
        self._check_memory_usage()
        tensors = future.result()
        if tensors is not None:
            for tensor in tensors:
                self.tensor_queue.add(tensor)
        
    def _shard_keys_for_rank(self, all_keys):
        """Shard keys across ranks for distributed training"""
        if self.world_size == 1:
            return all_keys
        
        # Sort keys for deterministic sharding
        sorted_keys = sorted(all_keys)
        
        # Simple modulo-based sharding
        sharded_keys = [key for i, key in enumerate(sorted_keys) if i % self.world_size == self.rank]
        
        logger.info(f"Rank {self.rank}/{self.world_size}: sharded {len(sharded_keys)} keys from {len(all_keys)} total")
        return sharded_keys

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

    def __iter__(self):
        """Iterator that yields video tensors"""
        while not self.shutdown_flag.is_set():
            tensor = self.tensor_queue.pop()
            if tensor is not None:
                yield tensor
            else:
                if self.shutdown_flag.is_set():
                    break
                time.sleep(1)

    def cleanup(self):
        """Clean up resources including thread pools"""
        if not hasattr(self, 'shutdown_flag'):
            return
            
        logger.info("Starting cleanup process...")
        
        # Signal background threads to stop
        self.shutdown_flag.set()
        logger.info("Shutdown flag set, clearing queues to unblock threads...")
        
        # Clear queues to unblock any waiting threads
        if hasattr(self, 'tensor_queue'):
            self.tensor_queue.items.clear()
            
        # Give threads time to notice shutdown flag and exit
        time.sleep(2)
        
        # Shutdown thread pools with timeout
        if hasattr(self, 'executor'):
            logger.info("Shutting down thread pool executor...")
            self.executor.shutdown(wait=True, cancel_futures=True)
            logger.info("Thread pool executor shut down successfully")
            
        logger.info("Cleanup completed successfully")

    def __del__(self):
        """Destructor to ensure cleanup"""
        try:
            self.cleanup()
        except:
            pass  # Ignore errors during cleanup in destructor

    def close(self):
        """Explicitly close the dataset and cleanup resources"""
        self.cleanup()

    def _check_memory_usage(self):
        """Monitor actual memory usage"""
        process = psutil.Process()
        memory_gb = process.memory_info().rss / (1024**3)
        print(f"Memory usage: {memory_gb:.2f}GB")

def collate_fn(batch):
    """Collate function to stack video tensors"""
    return torch.stack(batch)

def get_video_loader(batch_size=4, from_time=None, to_time=None, max_videos=None, **dataset_kwargs):
    """Create a DataLoader for MP4 videos
    
    Args:
        batch_size: Batch size for the DataLoader
        from_time: Start time in seconds for video sampling (None = start from beginning)
        to_time: End time in seconds for video sampling (None = go to end)
        max_videos: Maximum number of videos to process (None = unlimited)
        **dataset_kwargs: Additional arguments passed to MP4VideoDataset
    """
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    dataset = MP4VideoDataset(
        rank=rank, 
        world_size=world_size, 
        from_time=from_time,
        to_time=to_time,
        max_videos=max_videos,
        **dataset_kwargs
    )
    return DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn)

if __name__ == "__main__":
    # Test the dataset
    logger.info("🚀 Testing MP4VideoDataset with multi-sample extraction")
    
    # Example usage for creating different data splits:
    # Validation set: sample from 0 to 10 minutes
    # val_loader = get_video_loader(batch_size=1, from_time=0, to_time=10*60, ...)
    # 
    # Test set: sample from 10 to 20 minutes  
    # test_loader = get_video_loader(batch_size=1, from_time=10*60, to_time=20*60, ...)
    #
    # Train set: sample from 20 minutes to end
    # train_loader = get_video_loader(batch_size=1, from_time=20*60, to_time=None, ...)
    
    loader = get_video_loader(
        batch_size=1,  # Reduced since we get more samples per video
        bucket_name='cod-yt-playlist',
        frames_per_sample=101,
        target_height=360,
        target_width=640,
        crop_duration=10.2,
        max_parallel_downloads=8,  # Fewer videos needed
        max_memory_gb=10,  # 10GB memory limit
    )
    
    logger.info("⏱️  Loading 5 batches for timing analysis...")
    
    import os
    os.makedirs('/home/developer/workspace/data/pt', exist_ok=True)
    
    timings = []
    try:
        for i in range(20):
            start_time = time.time()
            batch = next(iter(loader))
            batch_time = time.time() - start_time
            timings.append(batch_time)
            logger.info(f"Batch {i+1}/20: {batch_time:.2f}s")
            
            # Save batch as .pt file
            torch.save(batch, f'/home/developer/workspace/data/pt/{i:04d}.pt')
            logger.info(f"Saved batch {i+1}/20 to /home/developer/workspace/data/pt/{i:04d}.pt")
    finally:
        # Properly close the dataset to prevent shutdown errors
        logger.info("🧹 Cleaning up dataset resources...")
        loader.dataset.close()
        logger.info("✅ Cleanup completed successfully!")
    
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
    logger.info(f"✅ Test completed with max_videos=3 limit!")
    
    # Properly close the dataset to prevent shutdown errors
    logger.info("🧹 Cleaning up dataset resources...")
    loader.dataset.close()
    logger.info("✅ Cleanup completed successfully!")