import fsspec
import threading
from dotenv import load_dotenv
import os

load_dotenv()

import torch
import random
from torch.utils.data import IterableDataset, DataLoader
import torch.distributed as dist
import time
from typing import Optional, List, Tuple, Any
import numpy as np
from collections import deque
import logging

logger = logging.getLogger(__name__)

class S3CoDLatentDataset(IterableDataset):
    def __init__(self, 
                 url,
                 window_length=33, 
                 rank=0, 
                 world_size=1, 
                 deterministic=True,
                 prefetch_size=100,
                 loop_forever=False):
        super().__init__()
        
        self.window = window_length
        self.rank = rank
        self.world_size = world_size
        self.url = url
        self.deterministic = deterministic
        self.prefetch_size = prefetch_size
        self.loop_forever = loop_forever

        # Setup fsspec filesystem
        self.fs = self._get_fs()

        # Get all file paths using glob
        self.file_paths = self._get_all_file_paths()
        
        # Shuffle if not deterministic
        if not deterministic:
            random.shuffle(self.file_paths)
        
        # Distribute files across ranks for distributed training
        self.file_paths = self.file_paths[rank::world_size]
        
        # Prefetch queue for processed data
        self.data_queue = deque(maxlen=prefetch_size)
        
        # Debug counters
        self.files_processed = 0
        self.samples_generated = 0
        self.samples_yielded = 0
        
        # Start prefetching thread
        self.prefetch_thread = threading.Thread(target=self._prefetch_worker, daemon=True)
        self.prefetch_thread.start()
        
        logger.info(f"Rank {rank}: Found {len(self.file_paths)} files, prefetch_size={prefetch_size}")

    def _get_fs(self):
        kwargs = {
            'key': os.getenv('AWS_ACCESS_KEY_ID'),
            'secret': os.getenv('AWS_SECRET_ACCESS_KEY'),
        }
        endpoint = os.getenv('AWS_ENDPOINT_URL_S3') 
        if endpoint:
            kwargs['endpoint_url'] = endpoint
        return fsspec.filesystem('s3', **{k:v for k,v in kwargs.items() if v})

    def _get_all_file_paths(self) -> List[str]:
        """Get all .pt file paths using glob"""
        try:
            # Use glob to find all .pt files in the bucket
            # Structure: s3://cod-yt-latent-pairs/vids_pt/train/0000/, 0001/, etc.
            # make sure url does not end with a /
            if self.url.endswith('/'):
                self.url = self.url[:-1]
            pattern = f"{self.url}/*/*.pt"
            files = self.fs.glob(pattern)
            # Convert to full paths and sort
            full_paths = sorted([f"s3://{file}" for file in files])
            if not self.deterministic:
                random.shuffle(full_paths)
            logger.info(f"Found {len(full_paths)} .pt files in bucket")
            return full_paths
            
        except Exception as e:
            logger.error(f"Error listing files: {e}")
            return []

    def load_tensor_file(self, file_path: str) -> Optional[torch.Tensor]:
        logger.debug(f"Loading file {file_path}")
        """Load a single tensor file from S3"""
        try:
            with self.fs.open(file_path, 'rb') as f:
                tensor = torch.load(f)
                logger.info(f"Successfully loaded {file_path}, tensor shape: {tensor.shape}, dtype: {tensor.dtype}")
                return tensor
        except Exception as e:
            logger.error(f"Error loading file {file_path}: {e}")
            return None

    def _process_file(self, file_path: str) -> List[torch.Tensor]:
        """Process a single file and return list of video samples"""
        samples = []
        try:
            # Load the tensor file
            tensor = self.load_tensor_file(file_path)
            if tensor is None:
                logger.warning(f"Skipping {file_path} - failed to load tensor")
                return samples

            # Assuming the tensor contains the video data with format [numframes, 3, 360, 640]
            if len(tensor.shape) == 4:  # [numframes, channels, height, width]
                num_frames = tensor.shape[0]
                logger.debug(f"Processing {file_path}: {num_frames} frames, shape {tensor.shape}")
                
                # Split tensor into non-overlapping window-sized chunks
                if num_frames >= self.window:
                    # Calculate how many complete windows we can extract
                    num_windows = num_frames // self.window
                    logger.debug(f"Extracting {num_windows} windows from {file_path}")
                    
                    for i in range(num_windows):
                        start_idx = i * self.window
                        end_idx = start_idx + self.window
                        
                        # Extract window
                        video_slice = tensor[start_idx:end_idx]
                        samples.append(video_slice)
                else:
                    logger.warning(f"Skipping {file_path} - insufficient frames ({num_frames} < {self.window})")
            else:
                logger.warning(f"Skipping {file_path} - unexpected tensor shape {tensor.shape}")

        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
        
        logger.debug(f"Generated {len(samples)} samples from {file_path}")
        return samples

    def _prefetch_worker(self):
        """Background thread to prefetch and process data"""
        file_index = 0
        done_once = False
        
        while True:
            # Check if we need more data
            queue_size = len(self.data_queue)
            if queue_size >= self.prefetch_size:
                logger.debug(f"Queue full ({queue_size}/{self.prefetch_size}), waiting...")
                time.sleep(0.1)
                continue
            
            # Process next file
            if file_index < len(self.file_paths):
                file_path = self.file_paths[file_index]
                logger.info(f"Prefetching file {file_index + 1}/{len(self.file_paths)}: {file_path}")
                
                start_time = time.time()
                samples = self._process_file(file_path)
                process_time = time.time() - start_time
                
                # Add samples to queue
                for sample in samples:
                    self.data_queue.append(sample)
                    self.samples_generated += 1
                    
                
                queue_size = len(self.data_queue)
                logger.info(f"Processed {file_path} in {process_time:.2f}s, generated {len(samples)} samples, queue size: {queue_size}/{self.prefetch_size}")
                
                self.files_processed += 1
                file_index += 1
            else:
                if not self.loop_forever:
                    logger.info(f"Completed processing all {len(self.file_paths)} files, not looping (loop_forever=False)")
                    break
                # If we've processed all files, restart from beginning
                logger.info(f"Completed processing all {len(self.file_paths)} files, restarting from beginning")
                if not self.deterministic:
                    # Shuffle files for next epoch
                    random.shuffle(self.file_paths)
                    logger.info("Shuffled file order for next epoch")
                file_index = 0
                self.files_processed = 0

    def __iter__(self):
        """Iterate over prefetched data samples"""
        logger.info(f"Starting iteration, initial queue size: {len(self.data_queue)}")
        empty_queue_count = 0
        yielded_any = False
        while True:
            if self.data_queue:
                sample = self.data_queue.popleft()
                self.samples_yielded += 1
                queue_size = len(self.data_queue)
                
                if self.samples_yielded % 100 == 0:  # Log every 100 samples
                    logger.info(f"Yielded {self.samples_yielded} samples, queue size: {queue_size}/{self.prefetch_size}, files processed: {self.files_processed}")
                
                yield sample
                yielded_any = True
                empty_queue_count = 0  # Reset counter
            else:
                # If not looping forever and prefetch thread is done, break
                if not self.loop_forever and self.files_processed >= len(self.file_paths):
                    logger.info("No more data to yield and loop_forever is False. Ending iteration.")
                    break
                empty_queue_count += 1
                if empty_queue_count % 100 == 0:  # Log every 100 empty checks
                    logger.warning(f"Queue empty for {empty_queue_count} consecutive checks, waiting...")
                # If no data available, wait a bit
                time.sleep(1.)

    def get_debug_stats(self):
        """Get current debug statistics"""
        return {
            'files_processed': self.files_processed,
            'samples_generated': self.samples_generated,
            'samples_yielded': self.samples_yielded,
            'queue_size': len(self.data_queue),
            'queue_capacity': self.prefetch_size,
            'total_files': len(self.file_paths)
        }

def collate_fn(batch):
    # batch is list of video tensors
    videos = torch.stack(batch)    # [b,n,c,h,w]
    return videos

def get_loader(batch_size, url, **data_kwargs):
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    ds = S3CoDLatentDataset(url=url, rank=rank, world_size=world_size, **data_kwargs)
    return DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn)

if __name__ == "__main__":
    import time
    
    # Configure logging for debugging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    loader = get_loader(10, window_length=5, deterministic=True, prefetch_size=30, url="s3://cod-yt-latent-pairs/vids_pt/train")

    start = time.time()
    batch = next(iter(loader))
    print(batch.shape)
    end = time.time()
    first_time = end - start
    
    # Get debug stats after first batch
    dataset = loader.dataset
    try:
        stats = dataset.get_debug_stats()  # type: ignore
        print(f"Debug stats after first batch: {stats}")
    except AttributeError:
        print("Debug stats not available")
    
    start = time.time()
    batch = next(iter(loader)) 
    print(batch.shape)
    end = time.time()
    second_time = end - start
    
    videos = batch
    print(f"Time to load first batch: {first_time:.2f}s")
    print(f"Time to load second batch: {second_time:.2f}s")
    print(f"Video shape: {videos.shape}")
    print(f"Video dtype: {videos.dtype}")
    
    # Get final debug stats
    try:
        stats = dataset.get_debug_stats()  # type: ignore
        print(f"Final debug stats: {stats}")
    except AttributeError:
        print("Debug stats not available")
