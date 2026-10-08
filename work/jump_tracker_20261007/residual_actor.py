"""One shared frozen backbone plus learned 3-output symmetric leg residual.
At deployment: same495-dimensional observation, no task ID or external phase logic.
"""
import torch
from torch import nn
class SharedResidualActor(nn.Module):
 def __init__(self,base,mean,std,projection,reference_only:bool=False,hidden_width:int=128):
  super().__init__();self.base=base;self.register_buffer('mean',mean);self.register_buffer('std',std);self.register_buffer('projection',projection)
  self.reference_only=reference_only;self.register_buffer('reference_indices',torch.tensor(list(range(58))+[160+67*k+j for k in range(5) for j in range(58)],dtype=torch.long));self.register_buffer('future_position_indices',torch.tensor([[160+67*k+58+j for j in range(3)] for k in range(5)],dtype=torch.long))
  self.head=nn.Sequential(nn.Linear(353 if reference_only else 495,hidden_width),nn.ELU(),nn.Linear(hidden_width,hidden_width),nn.ELU(),nn.Linear(hidden_width,3));nn.init.zeros_(self.head[-1].weight);nn.init.zeros_(self.head[-1].bias)
  for p in self.base.parameters():p.requires_grad_(False)
 def physical_residual(self,x):
  features=torch.clamp((x-self.mean)/(self.std+1e-8),-20.,20.)
  if self.reference_only:
   distances=torch.linalg.vector_norm(x[:,self.future_position_indices]-x[:,None,58:61],dim=-1)/.3
   features=torch.cat([features[:,self.reference_indices],distances],dim=-1)
  return .8*torch.tanh(self.head(features))
 def forward(self,x):return self.base(x)+self.physical_residual(x)@self.projection
