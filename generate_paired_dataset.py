#!/usr/bin/env python3
"""
Dataset generation script for WANDCAEPair model using Ray for distributed processing.
Processes video sequences and generates paired WAN/DCAE latent representations.
"""

import torch
import ray
from torch.utils.data import DataLoader
import fsspec
import webdataset as wds
import json
import typer
import os
from numpyencoder import NumpyEncoder
import logging
import io
from collections import defaultdict
from typing import List, Optional
import time
import numpy as np
from pathlib import Path

# Local imports
import sys
sys.path.append('./owl-wms')
sys.path.append('./owl-vaes')
sys.path.append('.')

from lat2lat.models.wan_dcae import WANDCAEPair
from lat2lat.data.tensor_s3 import get_loader

app = typer.Typer()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

def get_filesystem(url: str):
    """Get appropriate filesystem for URL."""
    if url.startswith("s3://"):
        return fsspec.filesystem("s3")
    else:
        return fsspec.filesystem("file")


def get_tar_writer(output_file: str):
    """Create a tar writer for the output file."""
    fs = get_filesystem(output_file)
    if output_file.startswith("s3://"):
        tar_fd = fs.open(output_file, "wb", s3={"profile": "writer"})
    else:
        tar_fd = fs.open(output_file, "wb")
    return wds.TarWriter(tar_fd)


@ray.remote(num_gpus=1)
class WANDCAEProcessor:
    """Ray actor for processing video sequences with WANDCAEPair model."""
    
    def __init__(self, gpu_id: int, data_url: str, output_path: str, batch_size = None, 
                 shard_size_mb: int = 100, sequence_length: int = 5, 
                 num_workers: int = 2, dtype_str: str = "bfloat16", compile: bool = False):
        self.gpu_id = gpu_id
        self.data_url = data_url
        self.output_path = output_path
        self.shard_size_mb = shard_size_mb
        self.sequence_length = sequence_length
        self.num_workers = num_workers
        self.dtype = getattr(torch, dtype_str)
        
        # Shard tracking
        self.current_shard_idx = 0
        self.current_folder_idx = 0
        self.max_tars_per_folder = 1000
        self.target_shard_size_bytes = shard_size_mb * 1024 * 1024
        
        # Current shard data
        self.current_shard_samples = []
        self.current_shard_size_bytes = 0
        
        # Initialize model
        self.model = WANDCAEPair(dtype=self.dtype)
        self.model.bfloat16()
        self.model.cuda(gpu_id)
        self.model.eval()
        if compile:
            self.model.compile()
        
        # Create output directory structure
        self.gpu_dir = os.path.join(output_path, f"{gpu_id:02d}")
        os.makedirs(self.gpu_dir, exist_ok=True)

        self.batch_size = batch_size
        
        logger.info(f"WANDCAEProcessor {gpu_id} initialized")
    
    def _get_shard_path(self) -> str:
        """Get path for current shard."""
        folder_path = os.path.join(self.gpu_dir, f"{self.current_folder_idx:04d}")
        os.makedirs(folder_path, exist_ok=True)
        return os.path.join(folder_path, f"{self.current_shard_idx:04d}.tar")
    
    def _write_current_shard(self):
        """Write current shard to disk."""
        if not self.current_shard_samples:
            return
            
        shard_path = self._get_shard_path()
        
        try:
            with get_tar_writer(shard_path) as sink:
                for i, sample in enumerate(self.current_shard_samples):
                    # Serialize tensors to bytes
                    wan_buffer = io.BytesIO()
                    torch.save(sample['wan_latent'], wan_buffer)
                    
                    dcae_buffer = io.BytesIO()
                    torch.save(sample['dcae_latent'], dcae_buffer)
                    
                    # Create samples for tar file with simple naming
                    wan_sample = {
                        '__key__': f"{i:04d}",
                        'wan.pt': wan_buffer.getvalue()
                    }
                    
                    dcae_sample = {
                        '__key__': f"{i:04d}",
                        'dcae.pt': dcae_buffer.getvalue()
                    }
                    
                    sink.write(wan_sample)
                    sink.write(dcae_sample)
            
            sample_count = len(self.current_shard_samples)
            size_mb = self.current_shard_size_bytes / 1024 / 1024
            logger.info(f"GPU {self.gpu_id}: Written shard {self.current_shard_idx} with {sample_count} samples ({size_mb:.1f} MB)")
            
            # Update shard tracking
            self.current_shard_idx += 1
            if self.current_shard_idx >= self.max_tars_per_folder:
                self.current_shard_idx = 0
                self.current_folder_idx += 1
                logger.info(f"GPU {self.gpu_id}: Moving to folder {self.current_folder_idx:04d}")
            
            # Reset current shard
            self.current_shard_samples = []
            self.current_shard_size_bytes = 0
            
        except Exception as e:
            logger.error(f"GPU {self.gpu_id}: Error writing shard: {e}")
    
    def _estimate_sample_size(self, sample: dict) -> int:
        """Estimate the size of a sample in bytes."""
        wan_tensor = sample['wan_latent']
        dcae_tensor = sample['dcae_latent']
        
        # Calculate tensor sizes in bytes
        wan_size = wan_tensor.numel() * wan_tensor.element_size()
        dcae_size = dcae_tensor.numel() * dcae_tensor.element_size()
        
        # Add some overhead for tar format and key names
        overhead = 512  # Reduced overhead since no metadata
        
        return wan_size + dcae_size + overhead
    
    def _add_sample_to_shard(self, sample: dict):
        """Add a sample to current shard, writing if full."""
        sample_size = self._estimate_sample_size(sample)
        
        # Write current shard if adding this sample would exceed size limit
        if (self.current_shard_size_bytes + sample_size) > self.target_shard_size_bytes and self.current_shard_samples:
            self._write_current_shard()
        
        # Add sample to current shard
        self.current_shard_samples.append(sample)
        self.current_shard_size_bytes += sample_size

    def get_max_batch_size(self, sequence_length: Optional[int] = None, height: int = 360, width: int = 640) -> int:
        """Determine maximum batch size for given input dimensions."""
        if sequence_length is None:
            sequence_length = self.sequence_length  # Use the actual sequence length!
        
        logger.info(f"GPU {self.gpu_id}: Starting batch size determination with sequence_length={sequence_length}, height={height}, width={width}")
        
        # Clear any existing GPU memory first
        torch.cuda.empty_cache()
        
        test_shape = (1, sequence_length, 3, height, width)
        
        # Start with a very conservative batch size and be more conservative overall
        batch_size = 1
        max_attempts = 8  # Reduced max attempts
        
        for attempt in range(max_attempts):
            try:
                test_input = (torch.randn(batch_size, *test_shape[1:])*255).to(torch.uint8)
                print(f"GPU {self.gpu_id}: TEST WAN input shape: {test_input.shape} type: {test_input.dtype}")
                with torch.no_grad():
                    x_wan, x_dcae = self.model.preprocess(test_input)
                    _ = self.model(x_wan.to(self.dtype).cuda(self.gpu_id), x_dcae.to(self.dtype).cuda(self.gpu_id))
                del test_input, x_wan, x_dcae
                torch.cuda.empty_cache()
                
                # If we get here, this batch size works
                logger.info(f"GPU {self.gpu_id}: Batch size {batch_size} works, trying larger...")
                
                # Double the batch size every step
                batch_size *= 2
                
                if batch_size > 512:  # Much lower maximum
                    optimal_batch_size = max(1, batch_size // 2)
                    self.batch_size = optimal_batch_size
                    logger.info(f"GPU {self.gpu_id}: Determined optimal batch size: {optimal_batch_size}")
                    return optimal_batch_size
            except RuntimeError as e:
                del test_input
                torch.cuda.empty_cache()
                if 'out of memory' in str(e):
                    optimal_batch_size = max(1, batch_size // 2)
                    self.batch_size = optimal_batch_size
                    logger.info(f"GPU {self.gpu_id}: OOM at batch_size={batch_size}, using optimal_batch_size={optimal_batch_size}")
                    return optimal_batch_size
                else:
                    logger.error(f"GPU {self.gpu_id}: Runtime error during batch size determination: {e}")
                    raise e
        
        # If we reach here, use the last successful batch size
        optimal_batch_size = max(1, batch_size // 2)
        self.batch_size = optimal_batch_size
        logger.info(f"GPU {self.gpu_id}: Reached max attempts, using optimal_batch_size={optimal_batch_size}")
        return optimal_batch_size
    
    def process_dataset(self, max_samples: Optional[int] = None) -> dict:
        """Process the entire dataset for this GPU."""
        try:
            # Ensure batch_size is set
            if self.batch_size is None:
                logger.info(f"GPU {self.gpu_id}: batch_size not set, determining optimal batch size...")
                try:
                    self.get_max_batch_size()
                except Exception as e:
                    logger.error(f"GPU {self.gpu_id}: Failed to determine batch size: {e}")
                    logger.info(f"GPU {self.gpu_id}: Using conservative default batch size of 1")
                    self.batch_size = 1
            
            # Create dataloader for this GPU
            logger.info(f"GPU {self.gpu_id}: Creating dataloader with batch_size={self.batch_size}")
            print(f"GPU {self.gpu_id}: Creating dataloader with batch_size={self.batch_size}")
            loader = get_loader(self.batch_size, url=self.data_url, window_length=self.sequence_length)
            logger.info(f"GPU {self.gpu_id}: Dataloader created successfully")
        except Exception as e:
            logger.error(f"GPU {self.gpu_id}: Failed to create dataloader: {e}")
            logger.error(f"GPU {self.gpu_id}: data_url={self.data_url}, batch_size={self.batch_size}")
            raise
        
        total_processed = 0
        total_batches = 0
        
        try:
            for batch_idx, batch in enumerate(loader):
                if max_samples and total_processed >= max_samples:
                    break
                
                try:
                    # Debug batch information
                    logger.info(f"GPU {self.gpu_id}: Processing batch {batch_idx}, batch type: {type(batch)}")
                    if isinstance(batch, torch.Tensor):
                        logger.info(f"GPU {self.gpu_id}: Batch shape: {batch.shape}, dtype: {batch.dtype}")
                    else:
                        logger.info(f"GPU {self.gpu_id}: Batch is not a tensor: {batch}")
                    
                    print(f"GPU {self.gpu_id}: Batch shape: {batch.shape}, dtype: {batch.dtype}")
                    # Process batch
                    with torch.no_grad():
                        # Process through model
                        try:
                            logger.info(f"GPU {self.gpu_id}: Starting model preprocessing...")
                            video_batch = batch  # Already in correct shape [batch_size, sequence_length, channels, height, width]
                            x_wan, x_dcae = self.model.preprocess(video_batch)
                            logger.info(f"GPU {self.gpu_id}: Preprocessing complete. WAN shape: {x_wan.shape if hasattr(x_wan, 'shape') else type(x_wan)}, DCAE shape: {x_dcae.shape if hasattr(x_dcae, 'shape') else type(x_dcae)}")
                        except Exception as e:
                            logger.error(f"GPU {self.gpu_id}: Error in model.preprocess(): {e}")
                            logger.error(f"GPU {self.gpu_id}: Input shape: {video_batch.shape}, dtype: {video_batch.dtype}")
                            import traceback
                            logger.error(f"GPU {self.gpu_id}: Preprocessing traceback:\n{traceback.format_exc()}")
                            raise
                        
                        try:
                            logger.info(f"GPU {self.gpu_id}: Starting model forward pass...")
                            logger.error(f"GPU {self.gpu_id}: TEST WAN input shape: {x_wan.shape} type: {x_wan.dtype}")
                            logger.error(f"GPU {self.gpu_id}: TEST DCAE input shape: {x_dcae.shape} type: {x_dcae.dtype}")
                            
                            # Clear cache before forward pass
                            torch.cuda.empty_cache()
                            
                            mean_wan, mean_dcae = self.model(x_wan.cuda(self.gpu_id), x_dcae.cuda(self.gpu_id))
                            logger.info(f"GPU {self.gpu_id}: Forward pass complete. WAN mean shape: {mean_wan.shape}, DCAE mean shape: {mean_dcae.shape}")
                        except Exception as e:
                            logger.error(f"GPU {self.gpu_id}: Error in model forward pass: {e}")
                            logger.error(f"GPU {self.gpu_id}: WAN input shape: {x_wan.shape if hasattr(x_wan, 'shape') else type(x_wan)}")
                            logger.error(f"GPU {self.gpu_id}: DCAE input shape: {x_dcae.shape if hasattr(x_dcae, 'shape') else type(x_dcae)}")
                            logger.error(f"GPU {self.gpu_id}: WAN input dtype: {x_wan.dtype if hasattr(x_wan, 'dtype') else type(x_wan)}")
                            logger.error(f"GPU {self.gpu_id}: DCAE input dtype: {x_dcae.dtype if hasattr(x_dcae, 'dtype') else type(x_dcae)}")
                            import traceback
                            logger.error(f"GPU {self.gpu_id}: Forward pass traceback:\n{traceback.format_exc()}")
                            raise
                        
                        # Move back to CPU for serialization
                        try:
                            mean_wan = mean_wan.cpu()
                            mean_dcae = mean_dcae.cpu()
                            logger.info(f"GPU {self.gpu_id}: Moved results to CPU")
                        except Exception as e:
                            logger.error(f"GPU {self.gpu_id}: Failed to move results to CPU: {e}")
                            raise
                        
                        # Create samples
                        try:
                            batch_size = video_batch.shape[0]
                            logger.info(f"GPU {self.gpu_id}: Creating {batch_size} samples...")
                            
                            for i in range(batch_size):
                                sample = {
                                    '__key__': f"gpu_{self.gpu_id:02d}_batch_{batch_idx:06d}_sample_{i:04d}",
                                    'wan_latent': mean_wan[i],
                                    'dcae_latent': mean_dcae[i]
                                }
                                
                                self._add_sample_to_shard(sample)
                                total_processed += 1
                            
                            logger.info(f"GPU {self.gpu_id}: Successfully created and added {batch_size} samples")
                        except Exception as e:
                            logger.error(f"GPU {self.gpu_id}: Error creating samples: {e}")
                            logger.error(f"GPU {self.gpu_id}: mean_wan shape: {mean_wan.shape}, mean_dcae shape: {mean_dcae.shape}")
                            import traceback
                            logger.error(f"GPU {self.gpu_id}: Sample creation traceback:\n{traceback.format_exc()}")
                            raise
                        
                        # Clean up memory after processing batch
                        del video_batch, x_wan, x_dcae, mean_wan, mean_dcae
                        torch.cuda.empty_cache()
                
                except Exception as e:
                    logger.error(f"GPU {self.gpu_id}: Error processing batch {batch_idx}: {e}")
                    
                    # If it's an OOM error, try with a smaller batch size
                    if 'out of memory' in str(e).lower() and self.batch_size is not None and self.batch_size > 1:
                        logger.warning(f"GPU {self.gpu_id}: OOM error, reducing batch size from {self.batch_size} to {max(1, self.batch_size - 1)}")
                        self.batch_size = max(1, self.batch_size - 1)
                        # Recreate dataloader with smaller batch size
                        try:
                            loader = get_loader(self.batch_size, url=self.data_url, window_length=self.sequence_length)
                            logger.info(f"GPU {self.gpu_id}: Recreated dataloader with batch_size={self.batch_size}")
                        except Exception as loader_error:
                            logger.error(f"GPU {self.gpu_id}: Failed to recreate dataloader: {loader_error}")
                    
                    # Continue to next batch instead of failing completely
                    continue
                
                total_batches += 1
                
                # Log progress
                if batch_idx % 10 == 0:
                    current_size_mb = self.current_shard_size_bytes / 1024 / 1024
                    logger.info(f"GPU {self.gpu_id}: Processed {total_processed} samples in {total_batches} batches "
                               f"(current shard: {current_size_mb:.1f} MB)")
        
        except KeyboardInterrupt:
            logger.info(f"GPU {self.gpu_id}: Interrupted by user")
        except Exception as e:
            logger.error(f"GPU {self.gpu_id}: Fatal error during processing: {e}")
            import traceback
            logger.error(f"GPU {self.gpu_id}: Full traceback:\n{traceback.format_exc()}")
            raise
        
        # Write final shard if there are remaining samples
        if self.current_shard_samples:
            self._write_current_shard()
        
        return {
            'gpu_id': self.gpu_id,
            'total_processed': total_processed,
            'total_batches': total_batches,
            'final_shard_idx': self.current_shard_idx,
            'final_folder_idx': self.current_folder_idx
        }


@app.command()
def generate_dataset(
    data_url: str = typer.Option(..., help="S3 URL containing video data"),
    output_path: str = typer.Option(..., help="Output path for generated dataset"),
    num_gpus: int = typer.Option(1, help="Number of GPUs to use"),
    batch_size: int = typer.Option(None, help="Batch size per GPU"),
    sequence_length: int = typer.Option(5, help="Number of frames per sequence"),
    shard_size_mb: int = typer.Option(100, help="Target size per shard in MB"),
    dtype: str = typer.Option("bfloat16", help="Data type for model"),
    num_workers: int = typer.Option(2, help="Number of dataloader workers per GPU"),
    max_samples: Optional[int] = typer.Option(None, help="Maximum number of samples to process per GPU"),
    compile: bool = typer.Option(False, help="Compile the model"),
):
    """Generate paired WAN/DCAE latent dataset using Ray for distributed processing."""
    
    # Initialize Ray
    if not ray.is_initialized():
        ray.init()
    
    logger.info(f"Starting dataset generation with {num_gpus} GPUs")
    logger.info(f"Each GPU will process data independently with batch_size={batch_size}")
    
    # Create output directory
    os.makedirs(output_path, exist_ok=True)
    
    # Initialize processors
    processors = []
    for gpu_id in range(num_gpus):
        processor = WANDCAEProcessor.remote(
            gpu_id=gpu_id,
            data_url=data_url,
            output_path=output_path,
            batch_size=batch_size,
            shard_size_mb=shard_size_mb,
            sequence_length=sequence_length,
            num_workers=num_workers,
            dtype_str=dtype,
            compile=compile
        )
        processors.append(processor)
    
    # Start processing
    # Call get_max_batch_size with explicit parameters to ensure consistency
    futures = [processor.get_max_batch_size.remote(sequence_length=sequence_length, height=360, width=640) for processor in processors]
    batch_sizes = ray.get(futures)
    print(f"Batch sizes: {batch_sizes}")
    logger.info(f"Batch sizes: {batch_sizes}")
    # Start processing on all GPUs
    logger.info("Starting processing on all GPUs...")
    futures = [processor.process_dataset.remote(max_samples) for processor in processors]
    
    # Wait for all processors to complete
    results = ray.get(futures)
    
    # Log results
    total_samples = sum(result['total_processed'] for result in results)
    logger.info(f"Dataset generation completed!")
    logger.info(f"Total samples processed: {total_samples}")
    
    for result in results:
        logger.info(f"GPU {result['gpu_id']}: {result['total_processed']} samples, "
                   f"{result['total_batches']} batches, "
                   f"final shard: {result['final_folder_idx']:04d}/{result['final_shard_idx']:04d}")
    
    # Cleanup
    ray.shutdown()





if __name__ == "__main__":
    app() 