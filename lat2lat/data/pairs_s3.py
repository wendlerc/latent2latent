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
import tarfile
import io

logger = logging.getLogger(__name__)

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

class LatentPairs(IterableDataset):
    def __init__(self, 
                 url,
                 rank=0, 
                 world_size=1, 
                 deterministic=True,
                 prefetch_size=100,
                 loop_forever=False,
                 file_share_max=20):
        super().__init__()
        self.rank = rank
        self.world_size = world_size
        self.url = url
        self.deterministic = deterministic
        self.prefetch_size = prefetch_size
        self.loop_forever = loop_forever
        self.file_share_max = file_share_max

        # Queue parameters
        self.max_tars = 2
        self.max_data = prefetch_size

        # Initialize queues
        self.tar_queue = RandomizedQueue()
        self.data_queue = RandomizedQueue()

        # Setup fsspec filesystem
        self.fs = self._get_fs()

        # Get all tar file paths using glob
        self.tar_paths = self._get_all_tar_paths()
        
        # Shuffle if not deterministic
        if not deterministic:
            random.shuffle(self.tar_paths)
        
        # Distribute files across ranks for distributed training
        self.tar_paths = self.tar_paths[rank::world_size]
        
        # Debug counters
        self.tars_processed = 0
        self.samples_generated = 0
        self.samples_yielded = 0
        
        # Start background threads
        self.tar_thread = threading.Thread(target=self._prefetch_tars_worker, daemon=True)
        self.data_thread = threading.Thread(target=self._process_data_worker, daemon=True)
        self.tar_thread.start()
        self.data_thread.start()
        
        logger.info(f"Rank {rank}: Found {len(self.tar_paths)} tar files, prefetch_size={prefetch_size}")

    def _get_fs(self):
        kwargs = {
            'key': os.getenv('AWS_ACCESS_KEY_ID'),
            'secret': os.getenv('AWS_SECRET_ACCESS_KEY'),
        }
        endpoint = os.getenv('AWS_ENDPOINT_URL_S3') 
        if endpoint:
            kwargs['endpoint_url'] = endpoint
        return fsspec.filesystem('s3', **{k:v for k,v in kwargs.items() if v})

    def _get_all_tar_paths(self) -> List[str]:
        """Get all .tar file paths using glob"""
        try:
            # make sure url does not end with a /
            if self.url.endswith('/'):
                self.url = self.url[:-1]
            
            # Use find to get all .tar files recursively
            files = self.fs.find(self.url, maxdepth=None, withdirs=False, detail=False)
            tar_files = [f for f in files if f.endswith('.tar')]
            
            # Convert to full paths and sort
            full_paths = sorted([f"s3://{file}" for file in tar_files])
            if not self.deterministic:
                random.shuffle(full_paths)
            logger.info(f"Found {len(full_paths)} .tar files in bucket")
            return full_paths
            
        except Exception as e:
            logger.error(f"Error listing tar files: {e}")
            return []

    def _prefetch_tars_worker(self):
        """Background thread to prefetch tar files"""
        tar_index = 0
        
        while True:
            # Check if we need more tars
            if len(self.tar_queue.items) < self.max_tars:
                if tar_index < len(self.tar_paths):
                    tar_path = self.tar_paths[tar_index]
                    logger.info(f"Prefetching tar {tar_index + 1}/{len(self.tar_paths)}: {tar_path}")
                    
                    try:
                        # Download tar directly to memory using fsspec
                        with self.fs.open(tar_path, 'rb') as f:
                            tar_data = f.read()
                        self.tar_queue.add(tar_data)
                        logger.info(f"Successfully downloaded {tar_path}")
                    except Exception as e:
                        logger.error(f"Error downloading tar {tar_path}: {e}")
                    
                    tar_index += 1
                else:
                    if not self.loop_forever:
                        logger.info(f"Completed downloading all {len(self.tar_paths)} tar files, not looping")
                        break
                    # If we've processed all files, restart from beginning
                    logger.info(f"Completed downloading all {len(self.tar_paths)} tar files, restarting from beginning")
                    if not self.deterministic:
                        random.shuffle(self.tar_paths)
                    tar_index = 0
            else:
                time.sleep(0.1)

    def _process_tensor_file(self, tar, base_name: str, suffix: str) -> Optional[torch.Tensor]:
        """Extract and load a tensor file from tar"""
        try:
            f = tar.extractfile(f"{base_name}.{suffix}.pt")
            if f is not None:
                tensor_data = f.read()
                tensor = torch.load(io.BytesIO(tensor_data))
                return tensor
        except Exception as e:
            logger.error(f"Error processing tensor file {base_name}.{suffix}.pt: {e}")
        return None

    def _process_data_worker(self):
        """Background thread to process tar data and extract samples"""
        while True:
            if len(self.data_queue.items) < self.max_data:
                tar_data = self.tar_queue.pop()
                if tar_data is None:
                    time.sleep(0.1)
                    continue

                try:
                    tar_file = io.BytesIO(tar_data)
                    with tarfile.open(fileobj=tar_file) as tar:
                        members = tar.getmembers()
                        base_names = set()
                        
                        # Get unique base names from .wan.pt files
                        for member in members:
                            if member.name.endswith('.wan.pt'):
                                base_names.add(member.name.split('.')[0])

                        for base_name in base_names:
                            # Load both wan and dcae tensors for this base name
                            wan_tensor = self._process_tensor_file(tar, base_name, "wan")
                            dcae_tensor = self._process_tensor_file(tar, base_name, "dcae")

                            if wan_tensor is not None and dcae_tensor is not None:
                                
                                # Sample multiple times if requested
                                for _ in range(self.file_share_max):
                                    if len(self.data_queue.items) >= self.max_data:
                                        break
                                    
                                    self.data_queue.add((wan_tensor, dcae_tensor))

                except Exception as e:
                    logger.error(f"Error processing tar: {e}")
            else:
                time.sleep(0.1)

    def __iter__(self):
        """Iterate over prefetched data samples"""
        logger.info(f"Starting iteration, initial queue size: {len(self.data_queue.items)}")
        empty_queue_count = 0
        
        while True:
            if self.data_queue.items:
                sample = self.data_queue.pop()
                if sample is not None:
                    self.samples_yielded += 1
                    queue_size = len(self.data_queue.items)
                    
                    if self.samples_yielded % 100 == 0:  # Log every 100 samples
                        logger.info(f"Yielded {self.samples_yielded} samples, queue size: {queue_size}/{self.max_data}")
                    
                    yield sample
                    empty_queue_count = 0  # Reset counter
                else:
                    empty_queue_count += 1
            else:
                # If not looping forever and all tars processed, break
                if not self.loop_forever and self.tars_processed >= len(self.tar_paths):
                    logger.info("No more data to yield and loop_forever is False. Ending iteration.")
                    break
                empty_queue_count += 1
                if empty_queue_count % 100 == 0:  # Log every 100 empty checks
                    logger.warning(f"Queue empty for {empty_queue_count} consecutive checks, waiting...")
                # If no data available, wait a bit
                time.sleep(0.1)

    def get_debug_stats(self):
        """Get current debug statistics"""
        return {
            'tars_processed': self.tars_processed,
            'samples_generated': self.samples_generated,
            'samples_yielded': self.samples_yielded,
            'queue_size': len(self.data_queue.items),
            'queue_capacity': self.max_data,
            'total_tars': len(self.tar_paths)
        }



def collate_fn_tar(batch):
    # batch is list of (wan_slice, dcae_slice) tuples
    wan_slices, dcae_slices = zip(*batch)
    wan_slices = torch.stack(wan_slices)    # [b,n,c,h,w]
    dcae_slices = torch.stack(dcae_slices)  # [b,n,c,h,w]
    return wan_slices, dcae_slices

def get_loader_tar(batch_size, url, **data_kwargs):
    if dist.is_initialized():
        rank = dist.get_rank()
        world_size = dist.get_world_size()
    else:
        rank = 0
        world_size = 1

    ds = LatentPairs(url=url, rank=rank, world_size=world_size, **data_kwargs)
    return DataLoader(ds, batch_size=batch_size, collate_fn=collate_fn_tar)

if __name__ == "__main__":
    import time
    
    # Configure logging for debugging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    loader = get_loader_tar(10, deterministic=True, prefetch_size=30, url="s3://cod-yt-latent-pairs/pairs/train2")

    start = time.time()
    wan, dcae = next(iter(loader))
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
    wan, dcae = next(iter(loader)) 
    end = time.time()
    second_time = end - start
    
    # print shapes and dtypes
    print(f"WAN shape: {wan.shape}")
    print(f"WAN dtype: {wan.dtype}")
    print(f"DCAE shape: {dcae.shape}")
    print(f"DCAE dtype: {dcae.dtype}")
    
    # Get final debug stats
    try:
        stats = dataset.get_debug_stats()  # type: ignore
        print(f"Final debug stats: {stats}")
    except AttributeError:
        print("Debug stats not available")
