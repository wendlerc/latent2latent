import torch
from torch import nn
import torch.nn.functional as F
from diffusers import AutoencoderKLWan
from owl_vaes.configs import ResNetConfig
from owl_vaes.models.dcae import DCAE, is_landscape
import einops as eo


class WANDCAEPair(nn.Module):
    def __init__(self, dtype = torch.bfloat16, landscape_size = (360, 640), square_size = (512, 512)):
        super().__init__()
        vae = AutoencoderKLWan.from_pretrained(
            "Wan-AI/Wan2.1-VACE-14B-diffusers",
            subfolder="vae",
            torch_dtype=torch.float16  # Optional: use half precision
        )
        self.wan_vae = vae
        self.wan_vae.decoder = None
        self.wan_vae.post_quant_conv = None
        self.dtype = dtype
        self.landscape_size = landscape_size
        self.square_size = square_size
        cfg = ResNetConfig(
            sample_size=[360,640],
            channels=3,
            latent_size=4,
            latent_channels=128,
            noise_decoder_inputs=0.0,
            ch_0=256,
            ch_max=2048,
            encoder_blocks_per_stage = [4, 4, 4, 4, 4, 4, 4],
            decoder_blocks_per_stage = [4, 4, 4, 4, 4, 4, 4]
        )
        dcae = DCAE(cfg)
        dcae.load_state_dict(torch.load("/home/developer/workspace/models/cod_128x.pt"))
        self.dcae = dcae
        self.dcae.decoder = None 
        self.dcae.eval()
        self.wan_vae.eval()
        self.to(dtype)


    #@torch.compile
    @torch.no_grad()
    def forward(self, x_wan, x_dcae):
        assert x_dcae.min() >= -1 and x_dcae.max() <= 1, "x_dcae must be in [-1,1]"
        assert len(x_dcae.shape) == 5, "x_dcae must be [B,T,C,H,W], B is batch size, T is number of frames, C is channels, H is height, W is width"
        # dcae expects [B,T,C,H,W]
        b,t,c,h,w = x_dcae.shape
        x_dcae = eo.rearrange(x_dcae, 'b t c h w -> (b t) c h w')
        assert self.dcae.training == False, "dcae must be in eval mode, otherwise it returns mean, logvar"
        mean_dcae = self.dcae.encoder(x_dcae)
        mean_dcae = eo.rearrange(mean_dcae, '(b t) c h w -> b t c h w', b=b, t=t)
        # wan expects [B,C,T,H,W]
        assert x_wan.min() >= -1 and x_wan.max() <= 1, "x_wan must be in [-1,1]"
        x_wan = eo.rearrange(x_wan, 'b t c h w -> b c t h w')
        assert (x_wan.shape[2]-1)%4 == 0, "wan does not compress the first frame but then the remaining ones by a factor of 4"
        out = self.wan_vae.encode(x_wan)
        mean_wan = out.latent_dist.mode()
        mean_wan = eo.rearrange(mean_wan, 'b c t h w -> b t c h w')
        assert mean_wan.shape[1] == 1 + (t-1)//4, "wan does not compress the first frame but then the remaining ones by a factor of 4"
        return mean_wan, mean_dcae


    def normalize(self, x):
        x = x.float()
        x = (x - x.min()) / (x.max() - x.min())
        x = 2 * x - 1
        x = torch.clamp(x, -1, 1)
        return x.to(self.dtype)

    def unnormalize(self, x):
        x = x.float()
        x = (x + 1) / 2
        x = x.clamp(0, 1)
        return x.to(self.dtype)
    def preprocess(self, x):
        """
        x: [B,T,C,H,W]
        normalizes x to [-1,1] and for wan squishes landscape to square
        returns x_wan, x_dcae
        """
        print(x.shape)
        b, t, c, h, w = x.shape
        
        if is_landscape((h, w)):
            # Reshape to [B*T,C,H,W] for interpolation
            x_flat = x.view(b * t, c, h, w)
            # use the best interpolation from torch.interpolate, e.g. lancosz
            x_wan = F.interpolate(x_flat, size=self.square_size, mode='bilinear', align_corners=False)
            x_wan = x_wan.clamp(0, 255)
            # Reshape back to [B,T,C,H,W]
            x_wan = x_wan.view(b, t, c, *self.square_size)
            
            if h == self.landscape_size[0] and w == self.landscape_size[1]:
                x_dcae = x
            else:
                x_dcae = F.interpolate(x_flat, size=self.landscape_size, mode='bilinear', align_corners=False)
                x_dcae = x_dcae.clamp(0, 255)
                x_dcae = x_dcae.view(b, t, c, *self.landscape_size)
        elif h == self.square_size[0] and w == self.square_size[1]:
            x_wan, x_dcae = x, x
        else:
            x_flat = x.view(b * t, c, h, w)
            x_wan = F.interpolate(x_flat, size=self.square_size, mode='bilinear', align_corners=False)
            x_wan = x_wan.clamp(0, 255)
            x_wan = x_wan.view(b, t, c, *self.square_size)
            x_dcae = F.interpolate(x_flat, size=self.square_size, mode='bilinear', align_corners=False)
            x_dcae = x_dcae.clamp(0, 255)
            x_dcae = x_dcae.view(b, t, c, *self.square_size)
        
        return self.normalize(x_wan), self.normalize(x_dcae)