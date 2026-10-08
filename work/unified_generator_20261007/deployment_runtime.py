"""One-file generation weights; no specialist checkpoint dependency at inference."""
from pathlib import Path
import sys,torch
from torch import nn
class EmbeddedNormalizer(nn.Module):
 def __init__(self,shape,eps=1e-12,disable=False):
  super().__init__();self.eps=eps;self.disable=disable;self.register_buffer('mean',torch.zeros(tuple(shape)));self.register_buffer('std',torch.ones(tuple(shape)))
 def __call__(self,x):return x if self.disable else (x-self.mean)/(self.std+self.eps)
 def inverse(self,x):return x if self.disable else x*self.std+self.mean

def load_one_checkpoint(path,device='cpu'):
 from common import pa
 import src.prepare
 from hydra.utils import instantiate
 from omegaconf import OmegaConf
 from control import RootControl
 pack=torch.load(path,map_location='cpu',weights_only=False)
 assert pack['kind']=='unified_full_frankenmotion_11'
 m=instantiate(OmegaConf.create(pack['diffusion_config']))
 m.denoiser=RootControl(m.denoiser)
 if pack.get('adapter_hidden_width',128)==512:
  from wide_adapter import WideTaskControl
  m.denoiser=WideTaskControl(m.denoiser)
 else:m.denoiser=pa.TemporalTaskControl(m.denoiser)
 m.load_state_dict(pack['state_dict'],strict=True)
 return m.to(device).eval()
