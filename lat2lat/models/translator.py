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


class LatentTranslator(nn.Module):
    """
    Translator that converts latents from 101x128x4x4 to 26x16x64x64.
    Uses the same residual blocks and upblocks as the DCAE architecture.
    More aggressive channel reduction.
    """
    def __init__(self, input_channels=128, output_channels=16, input_size=4, output_size=64):
        super().__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.input_size = input_size
        self.output_size = output_size
        
        # Upsample from 4x4 to 64x64 (16x upsampling)
        self.upsample_factor = output_size // input_size  # 16
        
        # Initial channel reduction
        self.conv_in = weight_norm(nn.Conv2d(input_channels, 64, 1, 1, 0))
        
        # Progressive upsampling using UpBlock pattern
        # Stage 1: 4x4 -> 8x8, channels: 64 -> 48
        self.stage1 = UpBlock(64, 48, num_res=2, total_blocks=2)
        
        # Stage 2: 8x8 -> 16x16, channels: 48 -> 32
        self.stage2 = UpBlock(48, 32, num_res=2, total_blocks=2)
        
        # Stage 3: 16x16 -> 32x32, channels: 32 -> 24
        self.stage3 = UpBlock(32, 24, num_res=2, total_blocks=2)

        # Stage 4: 32x32 -> 64x64, channels: 24 -> 16
        self.stage4 = UpBlock(24, 16, num_res=2, total_blocks=2)
        
        # Final refinement with SameBlock
        self.final = SameBlock(output_channels, output_channels, num_res=1, total_blocks=2)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, n, 128, 4, 4)
        Returns:
            Output tensor of shape (batch, 1 + (n-1)//4, 16, 64, 64)
        """
        # Initial channel reduction
        assert (x.shape[1] -1)%4 == 0, "Batch size -1 must be divisible by 4"
        b, n, c, h, w = x.shape
        x_flat = x.reshape(b*n, c, h, w)
        x = self.conv_in(x_flat)
        # Progressive upsampling stages (3 stages instead of 4)
        x = self.stage1(x)  # 4x4 -> 8x8, 64 -> 32 channels
        x = self.stage2(x)  # 8x8 -> 16x16, 32 -> 16 channels
        x = self.stage3(x)  # 16x16 -> 32x32, 16 -> 8 channels
        x = self.stage4(x)  # 32x32 -> 64x64, 8 -> 4 channels
        # Final refinement
        x = self.final(x)
        return x.reshape(b, n, -1, self.output_size, self.output_size)
    
    def translate_batch(self, x):
        """
        Translate a batch of latents from 101x128x4x4 to 26x16x64x64.
        Treats the first dimension separately as suggested.
        
        Args:
            x: Input tensor of shape (b, 101, 128, 4, 4)
        Returns:
            Output tensor of shape (b, 26, 16, 64, 64)
        """
        assert (x.shape[1] -1)%4 == 0, "Batch size -1 must be divisible by 4"
        translated = self.forward(x)
        b, n, c, h, w = translated.shape
        helper = translated[:, 1:].reshape(b, (n-1)//4, 4, c, h, w)
        reshaped = helper.mean(dim=2)
        return torch.cat([translated[:, 0:1], reshaped], dim=1)

    
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
        assert batch_output.shape == (2, 5, 16, 64, 64), f"Batch translation test failed: expected (17, 16, 64, 64), got {batch_output.shape}"

    print("LatentTranslator tests passed!")
    
if __name__ == "__main__":
    translator_test()
