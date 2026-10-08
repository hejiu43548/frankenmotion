"""Single-file FrankenMotion plus one shared parameter-command network.
Requires repository model code; no old control modules or weights are loaded.
Text encoder and SMPL are separate frontend/rendering assets, not controllers.
"""
import torch
from torch import nn
from unified_control import UnifiedControl,SharedCommands
from command_schema import FIELDS,INTENTS
class EmbeddedNormalizer(nn.Module):
 def __init__(self,shape,eps=1e-12,disable=False):
  super().__init__();self.eps=eps;self.disable=disable;self.register_buffer('mean',torch.zeros(tuple(shape)));self.register_buffer('std',torch.ones(tuple(shape)))
 def forward(self,x):return x if self.disable else (x-self.mean)/(self.std+self.eps)
 def inverse(self,x):return x if self.disable else x*self.std+self.mean

def load_one_checkpoint(path,device='cpu'):
 import src.prepare
 from hydra.utils import instantiate
 from omegaconf import OmegaConf
 pack=torch.load(path,map_location='cpu',weights_only=False)
 assert pack['kind']=='frankenmotion_all_commands_shared_v1'
 assert pack['fields']==FIELDS and pack['intents']==INTENTS
 model=instantiate(OmegaConf.create(pack['diffusion_config']))
 model.denoiser=UnifiedControl(model.denoiser,SharedCommands(pack['width']))
 model.load_state_dict(pack['state_dict'],strict=True)
 banned={'RootControl','TemporalTaskControl','WideTaskControl','GoalControl','ReachControl','ExitControl','LegacyTargets'}
 assert not any(type(m).__name__ in banned for m in model.modules())
 return model.to(device).eval()
