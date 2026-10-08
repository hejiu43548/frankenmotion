"""One shared wider MLP per denoiser layer, no experts or task dispatch."""
from common import pa,torch
from torch import nn
class WideTaskControl(pa.TemporalTaskControl):
 def __init__(self,root):
  super().__init__(root);dim=root.base.latent_dim
  self.residuals=nn.ModuleList([nn.Sequential(nn.Linear(dim,512),nn.SiLU(),nn.Linear(512,dim)) for _ in root.base.seqTransEncoder.layers])
  for layer in self.residuals:nn.init.zeros_(layer[-1].weight);nn.init.zeros_(layer[-1].bias)
 def expand_adapter(self,state):
  ref=self.adapter_state();out={}
  for k,target in ref.items():
   if k not in state:
    assert k.startswith('temporal.');out[k]=torch.zeros_like(target,device='cpu');continue
   value=state[k]
   if value.shape==target.shape:out[k]=value;continue
   out[k]=target.detach().cpu().clone()
   if k.endswith('.0.weight'):out[k][:value.shape[0]]=value
   elif k.endswith('.0.bias'):out[k][:value.shape[0]]=value
   elif k.endswith('.2.weight'):out[k].zero_();out[k][:,:value.shape[1]]=value
   else:raise ValueError(k)
  return out
 def load_adapter(self,state):
  result=self.load_state_dict(self.expand_adapter(state),strict=False);assert not result.unexpected_keys;assert all(k.startswith('root.') for k in result.missing_keys)
def enable():pa.core.TaskControl=WideTaskControl
