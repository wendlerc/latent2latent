#!/usr/bin/env python3
"""
Video dataset organization script using MP4VideoDataset.
Organizes video tensors into train/valid/test splits based on time ranges.

The dataloader will block when the upload queue is full, ensuring no data is lost
and the download rate is automatically throttled by the upload capacity.
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
import math

# Local imports
import sys
sys.path.append('.')

from lat2lat.data.mp4_video_dataset import get_video_loader

app = typer.Typer()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# Global upload queue and worker thread
MAX_QUEUE_SIZE = 20  # Queue size - dataloader will block when full to prevent data loss
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
            try:
                item = upload_queue.get(timeout=1)
            except:
                continue
                
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
                
        except Exception as e:
            logger.error(f"Upload worker error: {e}")
            continue

def ensure_upload_pool(num_workers=None, frames_per_sample=101, crop_duration=30.5):
    """Ensure the upload worker pool is running."""
    global upload_pool
    if upload_pool is None:
        if num_workers is None:
            # Default to 6 download workers if not specified
            num_workers = 8
        
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

def save_tensor(fs, tensor: torch.Tensor, file_path: str, frames_per_sample=101, crop_duration=30.5):
    """Save tensor to file using fsspec asynchronously."""
    try:
        # Convert tensor to bytes
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        buffer.seek(0)
        
        # Ensure upload pool is running
        ensure_upload_pool(frames_per_sample=frames_per_sample, crop_duration=crop_duration)
        
        # Block until there's space in the upload queue
        # This ensures the dataloader waits when uploads are slow
        while True:
            try:
                # Block indefinitely until there's space in the queue
                upload_queue.put((fs, buffer, file_path), block=True, timeout=None)
                
                # Free memory immediately after queuing
                del tensor
                gc.collect()
                
                return True
            except Full:
                # This should never happen with block=True and timeout=None,
                # but just in case, log and continue waiting
                logger.debug(f"Upload queue full, waiting for space... (queue size: {upload_queue.qsize()})")
                time.sleep(1)  # Brief pause before retrying
            
    except Exception as e:
        logger.error(f"Failed to prepare tensor for upload to {file_path}: {e}")
        return False

def get_filesystem(output_path: str):
    """Get appropriate filesystem for output path."""
    if output_path.startswith("s3://"):
        return fsspec.filesystem("s3")
    else:
        return fsspec.filesystem("file")


def create_folder_structure(fs, base_path: str, train_from: int, train_to: int):
    """Create the required folder structure based on train sample range."""
    folders = [
        "test/0000",
        "valid/0000"
    ]
    
    # Add train subfolders based on the sample range
    # Each subfolder contains 1000 samples, so compute subfolder indices
    start_subfolder = train_from // 1000
    end_subfolder = (train_to - 1) // 1000 + 1  # -1 to handle edge case, +1 for exclusive
    
    for i in range(start_subfolder, end_subfolder):
        folders.append(f"train/{i:04d}")
    
    for folder in folders:
        full_path = f"{base_path.rstrip('/')}/{folder}"
        try:
            fs.makedirs(full_path, exist_ok=True)
            logger.info(f"Created folder: {full_path}")
        except Exception as e:
            logger.warning(f"Could not create folder {full_path}: {e}")


def count_existing_samples(fs, split_name: str, base_path: str, train_from: int, train_to: int) -> int:
    """Count existing samples in a split folder, only within the sample range [train_from, train_to)."""
    total_samples = 0
    if split_name == "train":
        for sample_idx in range(train_from, train_to):
            subfolder_idx = sample_idx // 1000
            tensor_idx = sample_idx % 1000
            file_path = f"{base_path.rstrip('/')}/train/{subfolder_idx:04d}/{tensor_idx:04d}.pt"
            try:
                if fs.exists(file_path):
                    total_samples += 1
            except Exception as e:
                logger.debug(f"Error checking file {file_path}: {e}")
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
                 target_samples: int, base_path: str, fs, train_from: int, train_to: int, **loader_kwargs):
    """Process a data split (train/valid/test)."""
    if split_name == "train":
        # Count only the files in the relevant range
        existing_samples = count_existing_samples(fs, split_name, base_path, train_from, train_to)
        total_needed = train_to - train_from
        if existing_samples >= total_needed:
            logger.info(f"{split_name} split already has {existing_samples} samples in range [{train_from}, {train_to}) (target: {total_needed})")
            return existing_samples
        logger.info(f"Processing {split_name} split: samples {train_from} to {train_to} (exclusive)")
        logger.info(f"Found {existing_samples} existing samples in range, will collect {total_needed - existing_samples} more")
        
        crop_duration = loader_kwargs.get('crop_duration', 30.5)
        frames_per_sample = loader_kwargs.get('frames_per_sample', 101)
        nframes = crop_duration * 30
        max_videos = (total_needed - existing_samples)//(nframes//frames_per_sample) + 10
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
        sample_indices = [i for i in range(train_from, train_to)]
        # Only process missing files
        missing_indices = []
        for sample_idx in sample_indices:
            subfolder_idx = sample_idx // 1000
            tensor_idx = sample_idx % 1000
            file_path = f"{base_path.rstrip('/')}/train/{subfolder_idx:04d}/{tensor_idx:04d}.pt"
            if not fs.exists(file_path):
                missing_indices.append((sample_idx, subfolder_idx, tensor_idx, file_path))
        
        if not missing_indices:
            logger.info(f"No missing samples to process in range [{train_from}, {train_to})")
            return samples_collected
        
        try:
            loader_iter = iter(loader)
            for idx, (sample_idx, subfolder_idx, tensor_idx, file_path) in enumerate(missing_indices):
                try:
                    batch = next(loader_iter)
                except StopIteration:
                    logger.warning(f"Dataloader exhausted before all missing samples were generated.")
                    break
                # Remove batch dimension and move to CPU if needed
                tensor = batch.squeeze(0)
                if tensor.device.type != 'cpu':
                    tensor = tensor.cpu()
                # Save tensor
                if save_tensor(fs, tensor, file_path, frames_per_sample=frames_per_sample, crop_duration=crop_duration):
                    samples_collected += 1
                    if samples_collected % 100 == 0:
                        queue_size = upload_queue.qsize()
                        logger.info(f"{split_name}: {samples_collected}/{total_needed} samples collected (upload queue: {queue_size}/{MAX_QUEUE_SIZE})")
                        gc.collect()
                else:
                    logger.error(f"Failed to save sample {sample_idx} for {split_name} due to filesystem error")
                del batch
                del tensor
        except KeyboardInterrupt:
            logger.info(f"{split_name}: Interrupted by user at {samples_collected} samples")
        except Exception as e:
            logger.error(f"{split_name}: Error during processing: {e}")
            raise
        logger.info(f"{split_name}: Completed with {samples_collected} samples in range [{train_from}, {train_to})")
        return samples_collected
    else:
        # Original logic for test/valid
        existing_samples = count_existing_samples(fs, split_name, base_path, train_from, train_to)
        if existing_samples >= target_samples:
            logger.info(f"{split_name} split already has {existing_samples} samples (target: {target_samples})")
            return existing_samples
        logger.info(f"Processing {split_name} split: {from_time}s to {to_time}s")
        logger.info(f"Found {existing_samples} existing samples, will collect {target_samples - existing_samples} more")
        crop_duration = loader_kwargs.get('crop_duration', 30.5)
        frames_per_sample = loader_kwargs.get('frames_per_sample', 101)
        nframes = crop_duration * 30
        max_videos = (target_samples - existing_samples)//(nframes//frames_per_sample) + 10
        max_videos = None
        loader = get_video_loader(
            batch_size=1,
            from_time=from_time,
            to_time=to_time,
            max_videos=max_videos,
            **loader_kwargs
        )
        samples_collected = existing_samples
        samples_in_current_subfolder = existing_samples
        try:
            for batch_idx, batch in enumerate(loader):
                if samples_collected >= target_samples:
                    break
                tensor = batch.squeeze(0)
                if tensor.device.type != 'cpu':
                    tensor = tensor.cpu()
                subfolder_path = f"{base_path.rstrip('/')}/{split_name}/0000"
                tensor_idx = samples_collected
                filename = f"{tensor_idx:04d}.pt"
                file_path = f"{subfolder_path}/{filename}"
                if save_tensor(fs, tensor, file_path, frames_per_sample=frames_per_sample, crop_duration=crop_duration):
                    samples_collected += 1
                    samples_in_current_subfolder += 1
                    if samples_collected % 100 == 0:
                        queue_size = upload_queue.qsize()
                        logger.info(f"{split_name}: {samples_collected}/{target_samples} samples collected (upload queue: {queue_size}/{MAX_QUEUE_SIZE})")
                        gc.collect()
                else:
                    logger.error(f"Failed to save sample {samples_collected} for {split_name} due to filesystem error")
                del batch
                del tensor
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
    test_samples: int = typer.Option(1000, help="Number of test samples"),
    valid_samples: int = typer.Option(1000, help="Number of validation samples"),
    train_samples: int = typer.Option(20000, help="Number of training samples"),
    train_from: int = typer.Option(0, help="Starting train sample number (inclusive)"),
    train_to: int = typer.Option(20000, help="Ending train sample number (exclusive)"),
):
    """Organize video dataset into train/valid/test splits based on time ranges."""
    try:
        logger.info(f"Starting dataset organization to: {output_path}")
        logger.info(f"Target samples - test: {test_samples}, valid: {valid_samples}, train: {train_samples}")
        logger.info(f"Train range: samples {train_from} to {train_to} (exclusive)")
        
        # Get filesystem
        fs = get_filesystem(output_path)
        
        # Create folder structure
        create_folder_structure(fs, output_path, train_from, train_to)
        
        # Common loader kwargs
        loader_kwargs = {
            'bucket_name': bucket_name,
            'prefix': prefix,
            'frames_per_sample': frames_per_sample,
            'target_height': target_height,
            'target_width': target_width,
            'crop_duration': crop_duration,
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
            train_from=train_from,
            train_to=train_to,
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
            train_from=train_from,
            train_to=train_to,
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
            train_from=train_from,
            train_to=train_to,
            **loader_kwargs
        )
        
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
def process_train_range(
    output_path: str = typer.Option(..., help="Output path (local folder or s3://bucket/prefix)"),
    bucket_name: str = typer.Option('cod-yt-playlist', help="S3 bucket name for input videos"),
    prefix: str = typer.Option('', help="S3 prefix for input videos"),
    frames_per_sample: int = typer.Option(101, help="Number of frames per sample"),
    target_height: int = typer.Option(360, help="Target video height"),
    target_width: int = typer.Option(640, help="Target video width"),
    crop_duration: float = typer.Option(30.5, help="Duration of video crops in seconds"),
    train_samples: int = typer.Option(20000, help="Number of training samples"),
    train_from: int = typer.Option(..., help="Starting train sample number (inclusive)"),
    train_to: int = typer.Option(..., help="Ending train sample number (exclusive)"),
):
    """Process only a specific range of train subfolders."""
    try:
        logger.info(f"Processing train samples {train_from} to {train_to} (exclusive)")
        logger.info(f"Target train samples: {train_samples}")
        
        # Get filesystem
        fs = get_filesystem(output_path)
        
        # Create folder structure for the specified range
        create_folder_structure(fs, output_path, train_from, train_to)
        
        # Common loader kwargs
        loader_kwargs = {
            'bucket_name': bucket_name,
            'prefix': prefix,
            'frames_per_sample': frames_per_sample,
            'target_height': target_height,
            'target_width': target_width,
            'crop_duration': crop_duration,
        }
        
        # Process only train split for the specified range
        logger.info("=" * 50)
        logger.info("PROCESSING TRAIN SPLIT (SPECIFIED RANGE)")
        logger.info("=" * 50)
        
        result = process_split(
            split_name="train",
            from_time=20 * 60,  # 1200s
            to_time=None,       # end
            target_samples=train_samples,
            base_path=output_path,
            fs=fs,
            train_from=train_from,
            train_to=train_to,
            **loader_kwargs
        )
        
        # Clean up upload pool
        cleanup_upload_pool()
        
        # Log final results
        logger.info("=" * 50)
        logger.info("TRAIN RANGE PROCESSING COMPLETED")
        logger.info("=" * 50)
        logger.info(f"Train samples processed: {result}")
        logger.info(f"Range: samples {train_from} to {train_to} (exclusive)")
        logger.info(f"Dataset saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Error during train range processing: {e}")
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
        crop_duration=30.5,
    )
    
    logger.info("Test completed successfully!")


@app.command()
def verify_structure(
    dataset_path: str = typer.Option(..., help="Path to verify structure"),
    train_from: int = typer.Option(0, help="Starting train sample number (inclusive)"),
    train_to: int = typer.Option(20000, help="Ending train sample number (exclusive)"),
):
    """Verify the created dataset structure."""
    
    fs = get_filesystem(dataset_path)
    
    logger.info(f"Verifying dataset structure at: {dataset_path}")
    logger.info(f"Train range: subfolders {train_from} to {train_to} (exclusive)")
    
    # Expected structure
    expected_folders = ["test/0000", "valid/0000"]
    for i in range(train_from, train_to):
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