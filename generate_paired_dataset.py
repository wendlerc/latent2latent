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
from owl_vaes.data.local_cod import get_loader

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
    
    def __init__(self, gpu_id: int, data_root: str, output_path: str, batch_size = None, 
                 shard_size_mb: int = 100, sequence_length: int = 5, 
                 num_workers: int = 2, dtype_str: str = "bfloat16", compile: bool = False):
        self.gpu_id = gpu_id
        self.data_root = data_root
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
                for sample in self.current_shard_samples:
                    # Serialize tensors to bytes
                    wan_buffer = io.BytesIO()
                    torch.save(sample['wan_latent'], wan_buffer)
                    
                    dcae_buffer = io.BytesIO()
                    torch.save(sample['dcae_latent'], dcae_buffer)
                    
                    # Create sample for tar file
                    tar_sample = {
                        '__key__': sample['__key__'],
                        'wan_latent.pt': wan_buffer.getvalue(),
                        'dcae_latent.pt': dcae_buffer.getvalue(),
                        'metadata.json': json.dumps(sample['metadata'], cls=NumpyEncoder).encode('utf-8')
                    }
                    
                    sink.write(tar_sample)
            
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
        
        # Add metadata size (approximate)
        metadata_size = len(json.dumps(sample['metadata'], cls=NumpyEncoder).encode('utf-8'))
        
        # Add some overhead for tar format and key names
        overhead = 1024  # 1KB overhead per sample
        
        return wan_size + dcae_size + metadata_size + overhead
    
    def _add_sample_to_shard(self, sample: dict):
        """Add a sample to current shard, writing if full."""
        sample_size = self._estimate_sample_size(sample)
        
        # Write current shard if adding this sample would exceed size limit
        if (self.current_shard_size_bytes + sample_size) > self.target_shard_size_bytes and self.current_shard_samples:
            self._write_current_shard()
        
        # Add sample to current shard
        self.current_shard_samples.append(sample)
        self.current_shard_size_bytes += sample_size

    def get_max_batch_size(self, sequence_length: int = None, height: int = 360, width: int = 640) -> int:
        """Determine maximum batch size for given input dimensions."""
        if sequence_length is None:
            sequence_length = self.sequence_length  # Use the actual sequence length!
        
        test_shape = (1, sequence_length, 3, height, width)
        
        batch_size = 1
        while True:
            try:
                test_input = torch.randn(batch_size, *test_shape[1:]).to(self.dtype).cuda(self.gpu_id)
                with torch.no_grad():
                    x_wan, x_dcae = self.model.preprocess(test_input)
                    _ = self.model(x_wan, x_dcae)
                del test_input
                torch.cuda.empty_cache()
                batch_size *= 2
                if batch_size > 1024:
                    optimal_batch_size = max(1, batch_size // 2)
                    self.batch_size = optimal_batch_size
                    return optimal_batch_size
            except RuntimeError as e:
                if 'out of memory' in str(e):
                    optimal_batch_size = max(1, batch_size // 2)
                    self.batch_size = optimal_batch_size
                    return optimal_batch_size
                else:
                    raise e
    
    def process_dataset(self, max_samples: Optional[int] = None) -> dict:
        """Process the entire dataset for this GPU."""
        try:
            # Create dataloader for this GPU
            logger.info(f"GPU {self.gpu_id}: Creating dataloader with batch_size={self.batch_size}")
            print(f"GPU {self.gpu_id}: Creating dataloader with batch_size={self.batch_size}")
            loader = get_loader(self.batch_size*self.sequence_length, root=self.data_root)
            logger.info(f"GPU {self.gpu_id}: Dataloader created successfully")
        except Exception as e:
            logger.error(f"GPU {self.gpu_id}: Failed to create dataloader: {e}")
            logger.error(f"GPU {self.gpu_id}: data_root={self.data_root}, batch_size={self.batch_size}")
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
                            video_batch = batch.reshape(self.batch_size, self.sequence_length, *batch.shape[1:])
                            x_wan, x_dcae = self.model.preprocess(video_batch.bfloat16())
                            logger.info(f"GPU {self.gpu_id}: Preprocessing complete. WAN shape: {x_wan.shape if hasattr(x_wan, 'shape') else type(x_wan)}, DCAE shape: {x_dcae.shape if hasattr(x_dcae, 'shape') else type(x_dcae)}")
                        except Exception as e:
                            logger.error(f"GPU {self.gpu_id}: Error in model.preprocess(): {e}")
                            logger.error(f"GPU {self.gpu_id}: Input shape: {video_batch.shape}, dtype: {video_batch.dtype}")
                            import traceback
                            logger.error(f"GPU {self.gpu_id}: Preprocessing traceback:\n{traceback.format_exc()}")
                            raise
                        
                        try:
                            logger.info(f"GPU {self.gpu_id}: Starting model forward pass...")
                            mean_wan, mean_dcae = self.model(x_wan.cuda(self.gpu_id), x_dcae.cuda(self.gpu_id))
                            logger.info(f"GPU {self.gpu_id}: Forward pass complete. WAN mean shape: {mean_wan.shape}, DCAE mean shape: {mean_dcae.shape}")
                        except Exception as e:
                            logger.error(f"GPU {self.gpu_id}: Error in model forward pass: {e}")
                            logger.error(f"GPU {self.gpu_id}: WAN input shape: {x_wan.shape if hasattr(x_wan, 'shape') else type(x_wan)}")
                            logger.error(f"GPU {self.gpu_id}: DCAE input shape: {x_dcae.shape if hasattr(x_dcae, 'shape') else type(x_dcae)}")
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
                                    'dcae_latent': mean_dcae[i],
                                    'metadata': {
                                        'wan_shape': list(mean_wan[i].shape),
                                        'dcae_shape': list(mean_dcae[i].shape),
                                        'original_shape': list(video_batch[i].shape),
                                        'dtype': str(self.dtype),
                                        'gpu_id': self.gpu_id
                                    }
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
                
                except Exception as e:
                    logger.error(f"GPU {self.gpu_id}: Error processing batch {batch_idx}: {e}")
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
    data_root: str = typer.Option(..., help="Root directory containing video data"),
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
            data_root=data_root,
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
    futures = [processor.get_max_batch_size.remote(sequence_length) for processor in processors]
    batch_sizes = ray.get(futures)
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


@app.command()
def debug_components(
    data_root: str = typer.Option(..., help="Root directory containing video data"),
    dtype: str = typer.Option("bfloat16", help="Data type for model"),
):
    """Debug individual components to isolate issues."""
    
    logger.info("=== DEBUGGING COMPONENTS ===")
    
    # Test 1: Data Loading
    logger.info("1. Testing data loader...")
    try:
        loader = get_loader(2, root=data_root)  # Small batch size
        logger.info("✓ Data loader created successfully")
        
        batch = next(iter(loader))
        logger.info(f"✓ Got batch: type={type(batch)}, shape={getattr(batch, 'shape', 'no shape')}")
        
        if isinstance(batch, torch.Tensor):
            logger.info(f"  Batch details: shape={batch.shape}, dtype={batch.dtype}, min={batch.min():.3f}, max={batch.max():.3f}")
        else:
            logger.info(f"  Batch is not a tensor: {batch}")
            
    except Exception as e:
        logger.error(f"✗ Data loader failed: {e}")
        import traceback
        logger.error(f"Data loader traceback:\n{traceback.format_exc()}")
        return
    
    # Test 2: Model Creation
    logger.info("2. Testing model creation...")
    try:
        dtype_obj = getattr(torch, dtype)
        model = WANDCAEPair(dtype=dtype_obj)
        model.cuda(0)  # Use GPU 0
        model.eval()
        logger.info("✓ Model created and moved to GPU successfully")
    except Exception as e:
        logger.error(f"✗ Model creation failed: {e}")
        import traceback
        logger.error(f"Model creation traceback:\n{traceback.format_exc()}")
        return
    
    # Test 3: Model Preprocessing
    logger.info("3. Testing model preprocessing...")
    try:
        if isinstance(batch, torch.Tensor):
            video_batch = batch.cuda(0)
            logger.info(f"  Input to preprocess: shape={video_batch.shape}, dtype={video_batch.dtype}")
            
            with torch.no_grad():
                x_wan, x_dcae = model.preprocess(video_batch)
                logger.info(f"✓ Preprocessing successful")
                logger.info(f"  WAN output: type={type(x_wan)}, shape={getattr(x_wan, 'shape', 'no shape')}")
                logger.info(f"  DCAE output: type={type(x_dcae)}, shape={getattr(x_dcae, 'shape', 'no shape')}")
        else:
            logger.error("Cannot test preprocessing: batch is not a tensor")
            return
    except Exception as e:
        logger.error(f"✗ Model preprocessing failed: {e}")
        import traceback
        logger.error(f"Preprocessing traceback:\n{traceback.format_exc()}")
        return
    
    # Test 4: Model Forward Pass
    logger.info("4. Testing model forward pass...")
    try:
        with torch.no_grad():
            mean_wan, mean_dcae = model(x_wan, x_dcae)
            logger.info(f"✓ Forward pass successful")
            logger.info(f"  WAN mean: shape={mean_wan.shape}, dtype={mean_wan.dtype}")
            logger.info(f"  DCAE mean: shape={mean_dcae.shape}, dtype={mean_dcae.dtype}")
    except Exception as e:
        logger.error(f"✗ Model forward pass failed: {e}")
        import traceback
        logger.error(f"Forward pass traceback:\n{traceback.format_exc()}")
        return
    
    logger.info("=== ALL COMPONENTS WORKING ===")
    logger.info("If you're still getting stack() errors, they might be in the Ray distributed processing.")


@app.command()
def test_processing(
    data_root: str = typer.Option(..., help="Root directory containing video data"),
    num_samples: int = typer.Option(10, help="Number of samples to test per GPU"),
    num_gpus: int = typer.Option(1, help="Number of GPUs to test"),
    dtype: str = typer.Option("bfloat16", help="Data type for model"),
):
    """Test the processing pipeline with a small number of samples."""
    
    if not ray.is_initialized():
        ray.init()
    
    logger.info("Testing processing pipeline...")
    
    # Create temporary output directory
    import tempfile
    with tempfile.TemporaryDirectory() as temp_dir:
        # Initialize processors
        processors = []
        for gpu_id in range(num_gpus):
            processor = WANDCAEProcessor.remote(
                gpu_id=gpu_id,
                data_root=data_root,
                output_path=temp_dir,
                shard_size_mb=10,  # Small shards for testing
                sequence_length=5,
                num_workers=1,
                dtype_str=dtype
            )
            processors.append(processor)
        
        # Start processing
        futures = [processor.get_max_batch_size.remote(5) for processor in processors]
        batch_sizes = ray.get(futures)
        logger.info(f"Batch sizes: {batch_sizes}")

        # Start processing
        futures = [processor.process_dataset.remote(num_samples) for processor in processors]
        results = ray.get(futures)
        
        # Log results
        for result in results:
            logger.info(f"GPU {result['gpu_id']}: Successfully processed {result['total_processed']} samples")
    
    ray.shutdown()
    logger.info("Test completed successfully!")


if __name__ == "__main__":
    app() 