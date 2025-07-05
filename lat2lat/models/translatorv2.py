# python -m lat2lat.models.translator
import einops as eo
import torch
import torch.nn.functional as F
from torch import nn

from torch.nn.utils.parametrizations import weight_norm

import sys
import os
# Add the owl-vaes directory to the path
from ..nn.resnet import SameBlock, UpBlock

class LearnableTemporalAggregation(nn.Module):
    """
    Learnable temporal aggregation module that replaces simple averaging.
    Uses attention-based weighted combination of temporal frames.
    """
    def __init__(self, channels, group_size=4):
        super().__init__()
        self.group_size = group_size
        self.channels = channels
        
        # Attention mechanism for temporal aggregation
        self.temporal_attention = nn.Conv1d(channels, channels, kernel_size=group_size, stride=group_size)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (b, n, c, h, w) where n-1 is divisible by group_size
        Returns:
            Aggregated tensor of shape (b, 1 + (n-1)//group_size, c, h, w)
        """
        b, n, c, h, w = x.shape
        
        # Handle first frame separately
        first_frame = x[:, 0:1]
        
        # Reshape remaining frames into groups
        remaining_frames = x[:, 1:]
        
        y = eo.rearrange(remaining_frames, 'b n c h w -> (b h w) c n')
        y = self.temporal_attention(y)
        y = eo.rearrange(y, '(b h w) c n1 -> b n1 c h w', b=b,c=c,h=h,w=w,n1=n//4)
        return torch.cat([first_frame, y], dim=1)

class LatentTranslator(nn.Module):
    """
    Translator that converts latents from 101x128x4x4 to 26x16x64x64.
    Uses the same residual blocks and upblocks as the DCAE architecture.
    More aggressive channel reduction with learnable temporal aggregation and normalization.
    """
    def __init__(self, input_channels=128, output_channels=16, input_size=4, output_size=64, 
                 same_blocks_per_stage=None):
        super().__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.input_size = input_size
        self.output_size = output_size
        
        # Default same blocks configuration if not provided
        if same_blocks_per_stage is None:
            same_blocks_per_stage = [4, 4, 4, 4, 4]  # [input, stage1, stage2, stage3, stage4]
        
        assert len(same_blocks_per_stage) == 5, "same_blocks_per_stage must have 5 elements: [input, stage1, stage2, stage3, stage4]"
        self.same_blocks_per_stage = same_blocks_per_stage
        self.total_blocks = sum(same_blocks_per_stage)
        # Upsample from 4x4 to 64x64 (16x upsampling)
        self.upsample_factor = output_size // input_size  # 16
        
        # Initial channel reduction with normalization
        self.same_in = SameBlock(input_channels, input_channels, num_res=same_blocks_per_stage[0], total_blocks=self.total_blocks)
        self.norm_in = nn.GroupNorm(8, input_channels)
        self.conv_in = weight_norm(nn.Conv2d(input_channels, input_channels//2, kernel_size=3, stride=1, padding=1))
        
        # Progressive upsampling using UpBlock pattern with normalization
        # Stage 1: 4x4 -> 8x8, channels: 64 -> 48
        self.stage1 = UpBlock(input_channels//2, 3*input_channels//8, num_res=same_blocks_per_stage[1], total_blocks=self.total_blocks)
        self.norm1 = nn.GroupNorm(8, 3*input_channels//8)
        
        # Stage 2: 8x8 -> 16x16, channels: 48 -> 32
        self.stage2 = UpBlock(3*input_channels//8, input_channels//4, num_res=same_blocks_per_stage[2], total_blocks=self.total_blocks)
        self.norm2 = nn.GroupNorm(8, input_channels//4)
        
        # Stage 3: 16x16 -> 32x32, channels: 32 -> 24
        self.stage3 = UpBlock(input_channels//4, 3*input_channels//16, num_res=same_blocks_per_stage[3], total_blocks=self.total_blocks)
        self.norm3 = nn.GroupNorm(6, 3*input_channels//16)  # Adjusted for 24 channels
        
        # Stage 4: 32x32 -> 64x64, channels: 24 -> 16
        self.stage4 = UpBlock(3*input_channels//16, output_channels, num_res=same_blocks_per_stage[4], total_blocks=self.total_blocks)  
        self.norm4 = nn.GroupNorm(8, output_channels)
        
        # Final refinement with normalization
        self.final = weight_norm(nn.Conv2d(output_channels, output_channels, kernel_size=3, stride=1, padding=1))
        
        # Learnable temporal aggregation
        self.temporal_aggregator = LearnableTemporalAggregation(output_channels, group_size=4)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, n, 128, 4, 4)
        Returns:
            Output tensor of shape (batch, n, 16, 64, 64)
        """
        # Initial channel reduction
        assert (x.shape[1] -1)%4 == 0, f"{x.shape} Batch size -1 must be divisible by 4"
        b, n, c, h, w = x.shape
        x_flat = eo.rearrange(x, 'b n c h w -> (b n) c h w')
        
        # Forward pass with normalization
        x = self.same_in(x_flat)
        x = self.norm_in(x)
        x = self.conv_in(x)
        x = F.leaky_relu(x)
        
        # Progressive upsampling stages with normalization
        x = self.stage1(x)  # 4x4 -> 8x8, 64 -> 48 channels
        x = self.norm1(x)
        
        x = self.stage2(x)  # 8x8 -> 16x16, 48 -> 32 channels
        x = self.norm2(x)
        
        x = self.stage3(x)  # 16x16 -> 32x32, 32 -> 24 channels
        x = self.norm3(x)
        
        x = self.stage4(x)  # 32x32 -> 64x64, 24 -> 16 channels
        x = self.norm4(x)
        
        # Final refinement with normalization
        x = self.final(x)
        x = eo.rearrange(x, '(b n) c h w -> b n c h w', b=b,n=n)
        return x
    
    def translate_batch(self, x):
        """
        Translate a batch of latents from 101x128x4x4 to 26x16x64x64.
        Uses learnable temporal aggregation instead of simple averaging.
        
        Args:
            x: Input tensor of shape (b, 101, 128, 4, 4)
        Returns:
            Output tensor of shape (b, 26, 16, 64, 64)
        """
        assert (x.shape[1] -1)%4 == 0, f"{x.shape} Batch size -1 must be divisible by 4"
        translated = self.forward(x)
        
        # Apply learnable temporal aggregation
        return self.temporal_aggregator(translated)

    
def translator_test():
    """Test the LatentTranslator functionality"""
    translator = LatentTranslator()
    
    # Test single sample translation
    with torch.no_grad():
        # Test batch translation: 101x128x4x4 -> 26x16x64x64
        batch_input = torch.randn(2, 101, 128, 4, 4)
        batch_output = translator.translate_batch(batch_input)
        assert batch_output.shape == (2, 26, 16, 64, 64), f"Batch translation test failed: expected (2, 26, 16, 64, 64), got {batch_output.shape}"

        batch_input = torch.randn(2, 17, 128, 4, 4)
        batch_output = translator.translate_batch(batch_input)
        assert batch_output.shape == (2, 5, 16, 64, 64), f"Batch translation test failed: expected (2, 5, 16, 64, 64), got {batch_output.shape}"

    print("LatentTranslator tests passed!")
    
if __name__ == "__main__":
    translator_test()