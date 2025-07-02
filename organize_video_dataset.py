#!/usr/bin/env python3
"""
Video dataset organization script using MP4VideoDataset.
Organizes video tensors into train/valid/test splits based on time ranges.
"""

import torch
import fsspec
import typer
import os
import logging
from pathlib import Path
from typing import Optional
import time
import asyncio
from concurrent.futures import ThreadPoolExecutor
import io
from queue import Queue, Full
from threading import Thread, Event, Lock
import gc

# Local imports
import sys
sys.path.append('.')

from lat2lat.data.mp4_video_dataset import get_video_loader

app = typer.Typer()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Global upload queue and worker thread
MAX_QUEUE_SIZE = 50  # Limit number of tensors in memory
upload_queue = Queue(maxsize=MAX_QUEUE_SIZE)
upload_pool = None
stop_event = Event()
stats_lock = Lock()
upload_stats = {
    'completed': 0,
    'failed': 0,
    'last_log_time': time.time()
}

def upload_worker():
    """Background worker that processes tensor uploads from the queue."""
    while not stop_event.is_set() or not upload_queue.empty():
        try:
            # Get item from queue with timeout to allow checking stop_event
            item = upload_queue.get(timeout=1)
            if item is None:
                continue
                
            fs, buffer, file_path = item
            try:
                with fs.open(file_path, 'wb') as f:
                    f.write(buffer.getvalue())
                
                with stats_lock:
                    upload_stats['completed'] += 1
                    current_time = time.time()
                    if current_time - upload_stats['last_log_time'] > 10:
                        logger.info(f"Upload worker stats - Completed: {upload_stats['completed']}, "
                                  f"Failed: {upload_stats['failed']}, Queue size: {upload_queue.qsize()}")
                        upload_stats['last_log_time'] = current_time
                    
            except Exception as e:
                with stats_lock:
                    upload_stats['failed'] += 1
                logger.error(f"Failed to upload tensor to {file_path}: {e}")
            finally:
                buffer.close()  # Explicitly close buffer
                upload_queue.task_done()
                gc.collect()  # Help clean up memory
                
        except Exception:
            continue

def ensure_upload_pool(num_workers=6):
    """Ensure the upload worker pool is running."""
    global upload_pool
    if upload_pool is None:
        stop_event.clear()
        upload_pool = ThreadPoolExecutor(max_workers=num_workers, thread_name_prefix="TensorUploader")
        # Start the workers
        for _ in range(num_workers):
            upload_pool.submit(upload_worker)
        logger.info(f"Started upload pool with {num_workers} workers")

def cleanup_upload_pool():
    """Clean up the upload worker pool when done."""
    global upload_pool
    if upload_pool is not None:
        logger.info("Starting upload pool cleanup...")
        stop_event.set()
        
        # Wait for remaining uploads with progress updates
        start_time = time.time()
        remaining = upload_queue.qsize()
        if remaining > 0:
            logger.info(f"Waiting for {remaining} remaining uploads to complete...")
            
        last_progress_time = time.time()
        while not upload_queue.empty():
            current = upload_queue.qsize()
            current_time = time.time()
            
            # Log progress every 10 seconds
            if current_time - last_progress_time > 10:
                completed = remaining - current
                elapsed = current_time - start_time
                rate = completed / elapsed if elapsed > 0 else 0
                eta = current / rate if rate > 0 else "unknown"
                logger.info(f"Upload progress: {current} remaining, {rate:.2f} uploads/sec, ETA: {eta if isinstance(eta, str) else f'{eta:.0f}s'}")
                last_progress_time = current_time
            
            # Check for timeout after 10 minutes
            if (current_time - start_time) > 600:  # 10 minute timeout
                logger.warning("Upload cleanup timed out after 10 minutes!")
                break
                
            time.sleep(1)
            
        # Try to shutdown the pool
        try:
            upload_pool.shutdown(wait=True)
        except Exception as e:
            logger.error(f"Error shutting down upload pool: {e}")
        upload_pool = None
        
        # Report final stats
        final_remaining = upload_queue.qsize()
        if final_remaining > 0:
            logger.warning(f"Cleanup finished with {final_remaining} tensors still in queue")
        else:
            logger.info("All uploads completed successfully")
        
        # Force cleanup if anything remains
        while not upload_queue.empty():
            try:
                _, buffer, _ = upload_queue.get_nowait()
                buffer.close()
            except:
                pass
        gc.collect()

def save_tensor(fs, tensor: torch.Tensor, file_path: str):
    """Save tensor to file using fsspec asynchronously."""
    try:
        # Convert tensor to bytes
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        buffer.seek(0)
        
        # Ensure upload pool is running
        ensure_upload_pool()
        
        # Try to add to upload queue with timeout
        max_retries = 3
        retry_count = 0
        while retry_count < max_retries:
            try:
                # If queue is full, wait up to 60 seconds
                upload_queue.put((fs, buffer, file_path), timeout=60)
                
                # Free memory immediately after queuing
                del tensor
                gc.collect()
                
                return True
            except Full:
                retry_count += 1
                if retry_count < max_retries:
                    logger.warning(f"Upload queue full, retrying in 30 seconds... (attempt {retry_count}/{max_retries})")
                    time.sleep(30)  # Wait 30 seconds before retrying
                else:
                    logger.error(f"Upload queue full after {max_retries} retries, could not save tensor to {file_path}")
                    buffer.close()
                    return False
            
    except Exception as e:
        logger.error(f"Failed to prepare tensor for upload to {file_path}: {e}")
        return False

def get_filesystem(output_path: str):
    """Get appropriate filesystem for output path."""
    if output_path.startswith("s3://"):
        return fsspec.filesystem("s3")
    else:
        return fsspec.filesystem("file")


def create_folder_structure(fs, base_path: str):
    """Create the required folder structure."""
    folders = [
        "test/0000",
        "valid/0000"
    ]
    
    # Add train subfolders
    for i in range(20):
        folders.append(f"train/{i:04d}")
    
    for folder in folders:
        full_path = f"{base_path.rstrip('/')}/{folder}"
        try:
            fs.makedirs(full_path, exist_ok=True)
            logger.info(f"Created folder: {full_path}")
        except Exception as e:
            logger.warning(f"Could not create folder {full_path}: {e}")


def count_existing_samples(fs, split_name: str, base_path: str) -> int:
    """Count existing samples in a split folder."""
    total_samples = 0
    
    if split_name == "train":
        # For train: check all 20 subfolders
        for i in range(20):
            folder_path = f"{base_path.rstrip('/')}/{split_name}/{i:04d}"
            try:
                files = fs.ls(folder_path)
                pt_files = [f for f in files if f.endswith('.pt')]
                total_samples += len(pt_files)
            except Exception as e:
                logger.debug(f"Error checking folder {folder_path}: {e}")
    else:
        # For test/valid: check subfolder 0000
        folder_path = f"{base_path.rstrip('/')}/{split_name}/0000"
        try:
            files = fs.ls(folder_path)
            pt_files = [f for f in files if f.endswith('.pt')]
            total_samples = len(pt_files)
        except Exception as e:
            logger.debug(f"Error checking folder {folder_path}: {e}")
    
    return total_samples


def process_split(split_name: str, from_time: float, to_time: Optional[float], 
                 target_samples: int, base_path: str, fs, **loader_kwargs):
    """Process a data split (train/valid/test)."""
    # Check existing samples
    existing_samples = count_existing_samples(fs, split_name, base_path)
    if existing_samples >= target_samples:
        logger.info(f"{split_name} split already has {existing_samples} samples (target: {target_samples})")
        return existing_samples
    
    logger.info(f"Processing {split_name} split: {from_time}s to {to_time}s")
    logger.info(f"Found {existing_samples} existing samples, will collect {target_samples - existing_samples} more")
    
    crop_duration = loader_kwargs['crop_duration']
    nframes = crop_duration * 30
    frames_per_sample = loader_kwargs['frames_per_sample']
    max_videos = (target_samples - existing_samples)//(nframes//frames_per_sample) + 10
    max_videos = None
    
    # Create dataloader for this time range
    loader = get_video_loader(
        batch_size=1,
        from_time=from_time,
        to_time=to_time,
        max_videos=max_videos,
        **loader_kwargs
    )
    
    samples_collected = existing_samples
    if split_name == "train":
        current_subfolder = existing_samples // 1000
        samples_in_current_subfolder = existing_samples % 1000
    else:
        samples_in_current_subfolder = existing_samples
    
    try:
        for batch_idx, batch in enumerate(loader):
            if samples_collected >= target_samples:
                break
            
            # Remove batch dimension and move to CPU if needed
            tensor = batch.squeeze(0)
            if tensor.device.type != 'cpu':
                tensor = tensor.cpu()
            
            # Determine subfolder and filename
            if split_name == "train":
                # For train: distribute across 20 subfolders (0000-0019)
                subfolder_idx = current_subfolder
                tensor_idx = samples_in_current_subfolder
                subfolder_path = f"{base_path.rstrip('/')}/{split_name}/{subfolder_idx:04d}"
            else:
                # For test/valid: use subfolder 0000
                subfolder_path = f"{base_path.rstrip('/')}/{split_name}/0000"
                tensor_idx = samples_collected
            
            # Save tensor
            filename = f"{tensor_idx:04d}.pt"
            file_path = f"{subfolder_path}/{filename}"
            
            if save_tensor(fs, tensor, file_path):
                samples_collected += 1
                samples_in_current_subfolder += 1
                
                # For train split: move to next subfolder after 1000 samples
                if split_name == "train" and samples_in_current_subfolder >= 1000:
                    current_subfolder += 1
                    samples_in_current_subfolder = 0
                
                # Log progress
                if samples_collected % 100 == 0:
                    logger.info(f"{split_name}: {samples_collected}/{target_samples} samples collected")
                    # Force garbage collection periodically
                    gc.collect()
            else:
                logger.warning(f"Failed to save sample {samples_collected} for {split_name}")
            
            # Clear batch from memory
            del batch
            del tensor
            
            # If queue is getting full, give it time to process
            if upload_queue.qsize() > MAX_QUEUE_SIZE * 0.8:  # 80% full
                logger.info("Upload queue filling up, waiting for space...")
                wait_start = time.time()
                while upload_queue.qsize() > MAX_QUEUE_SIZE * 0.5:  # Wait until 50% full
                    time.sleep(1)
                    # Add timeout to prevent infinite wait
                    if time.time() - wait_start > 300:  # 5 minute timeout
                        logger.warning("Queue wait timeout reached, continuing...")
                        break
            
            # Log queue status periodically
            if batch_idx % 10 == 0:
                logger.info(f"Queue status - Size: {upload_queue.qsize()}/{MAX_QUEUE_SIZE}")
    
    except KeyboardInterrupt:
        logger.info(f"{split_name}: Interrupted by user at {samples_collected} samples")
    except Exception as e:
        logger.error(f"{split_name}: Error during processing: {e}")
        raise
    
    logger.info(f"{split_name}: Completed with {samples_collected} samples")
    return samples_collected


@app.command()
def organize_dataset(
    output_path: str = typer.Option(..., help="Output path (local folder or s3://bucket/prefix)"),
    bucket_name: str = typer.Option('cod-yt-playlist', help="S3 bucket name for input videos"),
    prefix: str = typer.Option('', help="S3 prefix for input videos"),
    frames_per_sample: int = typer.Option(101, help="Number of frames per sample"),
    target_height: int = typer.Option(360, help="Target video height"),
    target_width: int = typer.Option(640, help="Target video width"),
    crop_duration: float = typer.Option(30.5, help="Duration of video crops in seconds"),
    max_videos_queue: int = typer.Option(6, help="Max videos in processing queue"),
    max_tensors_queue: int = typer.Option(50, help="Max tensors in processing queue"),
    test_samples: int = typer.Option(1000, help="Number of test samples"),
    valid_samples: int = typer.Option(1000, help="Number of validation samples"),
    train_samples: int = typer.Option(20000, help="Number of training samples"),
):
    """Organize video dataset into train/valid/test splits based on time ranges."""
    try:
        logger.info(f"Starting dataset organization to: {output_path}")
        logger.info(f"Target samples - test: {test_samples}, valid: {valid_samples}, train: {train_samples}")
        
        # Get filesystem
        fs = get_filesystem(output_path)
        
        # Create folder structure
        create_folder_structure(fs, output_path)
        
        # Common loader kwargs
        loader_kwargs = {
            'bucket_name': bucket_name,
            'prefix': prefix,
            'frames_per_sample': frames_per_sample,
            'target_height': target_height,
            'target_width': target_width,
            'crop_duration': crop_duration,
            'max_videos_queue': max_videos_queue,
            'max_tensors_queue': max_tensors_queue,
        }
        
        # Process each split
        results = {}
        
        # Test split: 0s to 10*60s (0-600s)
        logger.info("=" * 50)
        logger.info("PROCESSING TEST SPLIT")
        logger.info("=" * 50)
        results['test'] = process_split(
            split_name="test",
            from_time=0,
            to_time=10 * 60,  # 600s
            target_samples=test_samples,
            base_path=output_path,
            fs=fs,
            **loader_kwargs
        )
        
        # Valid split: 10*60s to 20*60s (600-1200s)
        logger.info("=" * 50)
        logger.info("PROCESSING VALID SPLIT")
        logger.info("=" * 50)
        results['valid'] = process_split(
            split_name="valid",
            from_time=10 * 60,  # 600s
            to_time=20 * 60,    # 1200s
            target_samples=valid_samples,
            base_path=output_path,
            fs=fs,
            **loader_kwargs
        )
        
        # Train split: 20*60s to end (1200s+)
        logger.info("=" * 50)
        logger.info("PROCESSING TRAIN SPLIT")
        logger.info("=" * 50)
        results['train'] = process_split(
            split_name="train",
            from_time=20 * 60,  # 1200s
            to_time=None,       # end
            target_samples=train_samples,
            base_path=output_path,
            fs=fs,
            **loader_kwargs
        )
        
        # Wait for remaining uploads to complete
        if not upload_queue.empty():
            logger.info("Waiting for remaining uploads to complete...")
            upload_queue.join()
        
        # Clean up upload pool
        cleanup_upload_pool()
        
        # Log final results
        logger.info("=" * 50)
        logger.info("DATASET ORGANIZATION COMPLETED")
        logger.info("=" * 50)
        total_samples = sum(results.values())
        logger.info(f"Total samples processed: {total_samples}")
        
        for split_name, count in results.items():
            logger.info(f"{split_name}: {count} samples")
            if split_name == "train":
                subfolders = (count + 999) // 1000  # Ceiling division
                logger.info(f"  Distributed across {subfolders} subfolders")
        
        logger.info(f"Dataset saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error during dataset organization: {e}")
        cleanup_upload_pool()
        raise


@app.command()
def test_setup(
    output_path: str = typer.Option(..., help="Output path to test"),
    bucket_name: str = typer.Option('cod-yt-playlist', help="S3 bucket name"),
    test_samples: int = typer.Option(5, help="Number of test samples per split"),
):
    """Test the setup with a small number of samples."""
    
    logger.info("Testing dataset organization setup...")
    
    # Test with very small numbers
    organize_dataset(
        output_path=output_path,
        bucket_name=bucket_name,
        test_samples=test_samples,
        valid_samples=test_samples,
        train_samples=test_samples,
        crop_duration=10.2,
        max_videos_queue=2,
        max_tensors_queue=10
    )
    
    logger.info("Test completed successfully!")


@app.command()
def verify_structure(
    dataset_path: str = typer.Option(..., help="Path to verify structure"),
):
    """Verify the created dataset structure."""
    
    fs = get_filesystem(dataset_path)
    
    logger.info(f"Verifying dataset structure at: {dataset_path}")
    
    # Expected structure
    expected_folders = ["test/0000", "valid/0000"]
    for i in range(20):
        expected_folders.append(f"train/{i:04d}")
    
    for folder in expected_folders:
        folder_path = f"{dataset_path.rstrip('/')}/{folder}"
        try:
            files = fs.ls(folder_path)
            pt_files = [f for f in files if f.endswith('.pt')]
            logger.info(f"{folder}: {len(pt_files)} .pt files")
            
            if len(pt_files) > 0:
                # Check a sample file
                sample_file = pt_files[0]
                try:
                    with fs.open(sample_file, 'rb') as f:
                        tensor = torch.load(f, map_location='cpu')
                    logger.info(f"  Sample tensor shape: {tensor.shape}, dtype: {tensor.dtype}")
                except Exception as e:
                    logger.warning(f"  Could not load sample tensor: {e}")
                    
        except Exception as e:
            logger.error(f"{folder}: Error accessing folder: {e}")
    
    logger.info("Structure verification completed")


if __name__ == "__main__":
    app() 