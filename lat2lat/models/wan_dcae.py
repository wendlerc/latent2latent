import torch
from torch import nn
from diffusers import AutoencoderKLWan
from owl_vaes.configs import ResNetConfig
from owl_vaes.models.dcae import DCAE
import einops as eo

class WANDCAEPair(nn.Module):
    def __init__(self, dtype = torch.bfloat16):
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
    def forward(self, x):
        assert x.min() >= -1 and x.max() <= 1, "x must be in [-1,1]"
        assert len(x.shape) == 5, "x must be [B,T,C,H,W], B is batch size, T is number of frames, C is channels, H is height, W is width"
        # dcae expects [B,T,C,H,W]
        b,t,c,h,w = x.shape
        x_dcae = eo.rearrange(x, 'b t c h w -> (b t) c h w')
        assert self.dcae.training == False, "dcae must be in eval mode, otherwise it returns mean, logvar"
        mean_dcae = self.dcae.encoder(x_dcae)
        mean_dcae = eo.rearrange(mean_dcae, '(b t) c h w -> b t c h w', b=b, t=t)
        # wan expects [B,C,T,H,W]
        x_wan = eo.rearrange(x, 'b t c h w -> b c t h w')
        assert (x_wan.shape[2]-1)%4 == 0, "wan does not compress the first frame but then the remaining ones by a factor of 4"
        out = self.wan_vae.encode(x_wan)
        mean_wan = out.latent_dist.mode()
        mean_wan = eo.rearrange(mean_wan, 'b c t h w -> b t c h w')
        assert mean_wan.shape[1] == 1 + (t-1)//4, "wan does not compress the first frame but then the remaining ones by a factor of 4"
        return mean_wan, mean_dcae

    def preprocess(self, x):
        # normalize to [-1, 1]
        x = x.float()
        x = (x - x.min()) / (x.max() - x.min())
        x = 2 * x - 1
        x = torch.clamp(x, -1, 1)
        return x.to(self.dtype)
    
    def postprocess(self, x):
        x = x.float()
        x = (x + 1) / 2
        x = x.clamp(0, 1)
        return x.to(self.dtype)