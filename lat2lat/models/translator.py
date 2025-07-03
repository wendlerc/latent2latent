import einops as eo
import torch
import torch.nn.functional as F
from torch import nn

from torch.nn.utils.parametrizations import weight_norm

import sys
import os
# Add the owl-vaes directory to the path
current_dir = os.path.dirname(os.path.abspath(__file__))
owl_vaes_path = os.path.join(current_dir, '..', '..', 'owl-vaes')
sys.path.insert(0, owl_vaes_path)

from owl_vaes.nn.resnet import SameBlock, UpBlock


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
        
        # Progressive upsampling using UpBlock pattern (more aggressive)
        # Stage 1: 4x4 -> 8x8, channels: 64 -> 32
        self.stage1 = UpBlock(64, 32, num_res=2, total_blocks=2)
        
        # Stage 2: 8x8 -> 16x16, channels: 32 -> 16
        self.stage2 = UpBlock(32, 16, num_res=2, total_blocks=2)
        
        # Stage 3: 16x16 -> 32x32, channels: 16 -> 8
        self.stage3 = UpBlock(16, 8, num_res=2, total_blocks=2)
        
        # Final refinement with SameBlock
        self.final = SameBlock(output_channels, output_channels, num_res=1, total_blocks=2)
        
    def forward(self, x):
        """
        Args:
            x: Input tensor of shape (batch, 128, 4, 4)
        Returns:
            Output tensor of shape (batch, 16, 64, 64)
        """
        # Initial channel reduction
        x = self.conv_in(x)
        
        # Progressive upsampling stages (3 stages instead of 4)
        x = self.stage1(x)  # 4x4 -> 8x8, 64 -> 32 channels
        x = self.stage2(x)  # 8x8 -> 16x16, 32 -> 16 channels
        x = self.stage3(x)  # 16x16 -> 32x32, 16 -> 8 channels
        
        # Need to add one more upsampling stage to get to 64x64
        # Since we removed stage4, we need to handle the final upsampling
        x = F.interpolate(x, scale_factor=2, mode='bicubic', align_corners=False)  # 32x32 -> 64x64
        
        # Channel expansion back to output_channels
        x = weight_norm(nn.Conv2d(8, self.output_channels, 1, 1, 0))(x)
        
        # Final refinement
        x = self.final(x)
        
        return x
    
    def translate_batch(self, x):
        """
        Translate a batch of latents from 101x128x4x4 to 26x16x64x64.
        Treats the first dimension separately as suggested.
        
        Args:
            x: Input tensor of shape (101, 128, 4, 4)
        Returns:
            Output tensor of shape (26, 16, 64, 64)
        """
        assert (x.shape[0] -1)%4 == 0, "Batch size -1 must be divisible by 4"
        translated = self.forward(x)
        n, c, h, w = translated.shape
        helper = translated[1:].reshape((n-1)//4, 4, c, h, w)
        reshaped = helper.mean(dim=1)
        return torch.cat([translated[0:1], reshaped], dim=0)

    
def translator_test():
    """Test the LatentTranslator functionality"""
    translator = LatentTranslator()
    
    # Test single sample translation
    with torch.no_grad():
        # Test batch translation: 101x128x4x4 -> 26x16x64x64
        batch_input = torch.randn(101, 128, 4, 4)
        batch_output = translator.translate_batch(batch_input)
        assert batch_output.shape == (26, 16, 64, 64), f"Batch translation test failed: expected (26, 16, 64, 64), got {batch_output.shape}"

        batch_input = torch.randn(17, 128, 4, 4)
        batch_output = translator.translate_batch(batch_input)
        assert batch_output.shape == (5, 16, 64, 64), f"Batch translation test failed: expected (17, 16, 64, 64), got {batch_output.shape}"

    print("LatentTranslator tests passed!")
    
if __name__ == "__main__":
    translator_test()
