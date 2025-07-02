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

# Local imports
import sys
sys.path.append('.')

from lat2lat.data.mp4_video_dataset import get_video_loader

app = typer.Typer()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def get_filesystem(output_path: str):
    """Get appropriate filesystem for output path."""
    if output_path.startswith("s3://"):
        return fsspec.filesystem("s3")
    else:
        return fsspec.filesystem("file")


def save_tensor(fs, tensor: torch.Tensor, file_path: str):
    """Save tensor to file using fsspec."""
    try:
        # Convert tensor to bytes
        import io
        buffer = io.BytesIO()
        torch.save(tensor, buffer)
        buffer.seek(0)
        
        # Write to file
        with fs.open(file_path, 'wb') as f:
            f.write(buffer.getvalue())
        
        logger.debug(f"Saved tensor to {file_path}")
        return True
    except Exception as e:
        logger.error(f"Failed to save tensor to {file_path}: {e}")
        return False


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
            
            # Remove batch dimension: [1, 101, 3, 360, 640] -> [101, 3, 360, 640]
            tensor = batch.squeeze(0)
            
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
            else:
                logger.warning(f"Failed to save sample {samples_collected} for {split_name}")
    
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
    crop_duration: float = typer.Option(61, help="Duration of video crops in seconds"),
    max_videos_queue: int = typer.Option(6, help="Max videos in processing queue"),
    max_tensors_queue: int = typer.Option(50, help="Max tensors in processing queue"),
    test_samples: int = typer.Option(1000, help="Number of test samples"),
    valid_samples: int = typer.Option(1000, help="Number of validation samples"),
    train_samples: int = typer.Option(20000, help="Number of training samples"),
):
    """Organize video dataset into train/valid/test splits based on time ranges."""
    
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