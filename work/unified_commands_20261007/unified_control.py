"""Deployment control: ONE shared encoder, fixed dense heads, no legacy modules."""
import torch
from torch import nn
from command_schema import NF,availability
class SharedCommands(nn.Module):
 def __init__(self,width=512):
  super().__init__();self.width=width;self.encoder=nn.Sequential(nn.Linear(NF,width),nn.SiLU(),nn.Linear(width,width),nn.SiLU());self.residuals=nn.ModuleList([nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,512)) for _ in range(4)]);self.output=nn.Sequential(nn.Linear(width,width),nn.SiLU(),nn.Linear(width,205))
 def forward(self,c):
  h=self.encoder(c);gate=availability(c);return torch.stack([m(h) for m in self.residuals],-2)*gate[...,None],self.output(h)*gate
class UnifiedControl(nn.Module):
 def __init__(self,base,controller):
  super().__init__();self.base=base;self.base.requires_grad_(False);self.controller=controller;self.command_features=None;self.static_residuals=None;self.cached=None;self.handles=[layer.register_forward_hook(self.hook(i)) for i,layer in enumerate(base.seqTransEncoder.layers)]
 def hook(self,i):
  def add(module,args,out):
   if self.cached is None:return out
   z=self.cached[0][...,i,:];return out+torch.nn.functional.pad(z,(0,0,out.shape[1]-z.shape[1],0))
  return add
 def forward(self,x,y,t,tf=None):
  self.cached=self.static_residuals if self.static_residuals is not None else (None if self.command_features is None else self.controller(self.command_features))
  try:
   out=self.base(x,y,t,tf)
   if self.cached is not None:out=out+torch.nn.functional.pad(self.cached[1],(0,out.shape[-1]-205))
   return out
  finally:self.cached=None
 def train(self,mode=True):super().train(mode);self.base.eval();return self

def load_base(controller,device='cpu'):
 from uc_common import B
 from deployment_runtime import load_one_checkpoint
 m=load_one_checkpoint(B/'current_full.pt');old=m.denoiser;root=old.root;base=root.base
 for h in old.handles+root._handles:h.remove()
 m.denoiser=UnifiedControl(base,controller);return m.to(device).eval()
