"""TRAINING / BASELINE ONLY. Frozen old controls expressed as residual targets."""
from uc_common import *
class Basic(nn.Module):
 def __init__(self,state):
  super().__init__();din=state['encoder.0.weight'].shape[1];width=state['residuals.0.0.weight'].shape[0];self.encoder=nn.Sequential(nn.Linear(din,128),nn.SiLU(),nn.Linear(128,512));self.residuals=nn.ModuleList([nn.Sequential(nn.Linear(512,width),nn.SiLU(),nn.Linear(width,512)) for _ in range(4)])
  if 'task.weight' in state:self.task=nn.Embedding(11,32)
  if 'temporal.weight' in state:self.temporal=nn.Linear(17,512)
  if 'output_head.0.weight' in state:self.output_head=nn.Sequential(nn.Linear(11,128),nn.SiLU(),nn.Linear(128,205))
  self.load_state_dict(state,strict=True);self.requires_grad_(False)
 def project(self,h):return torch.stack([r(h) for r in self.residuals],-2)
class LegacyTargets(nn.Module):
 def __init__(self):
  super().__init__();p=torch.load(B/'current_full.pt',map_location='cpu',weights_only=False)['state_dict'];a={k[len('denoiser.'):]:v for k,v in p.items() if k.startswith('denoiser.') and not k.startswith('denoiser.root.')};self.current=Basic(a)
  for name in ['root','physical','goal','reach','exit']:setattr(self,name,Basic(torch.load(B/(name+'.pt'),map_location='cpu',weights_only=False)['adapter']))
  embedded={k[len('denoiser.root.'):]:v for k,v in p.items() if k.startswith('denoiser.root.') and not k.startswith('denoiser.root.base.')}
  assert all(torch.equal(v,self.root.state_dict()[k]) for k,v in embedded.items());self.eval()
 @torch.no_grad()
 def forward(self,c):
  shape=c.shape[:-1];x=c.reshape(-1,NF);tid=x[:,:12].argmax(-1);intent=x[:,12:16].argmax(-1);phase=x[:,28];ft=x[:,28:45];r=x.new_zeros(len(x),4,512);o=x.new_zeros(len(x),205)
  rc=x[:,18:22];r+=self.root.project(self.root.encoder(rc))*rc[:,2:].amax(-1)[:,None,None]
  rr=x.new_tensor(RANGES)[tid.clamp_max(10)];cmd=(x[:,16]+1)*.5*(rr[:,1]-rr[:,0])+rr[:,0]
  for module,mask in [(self.current,(intent==0)&(x[:,17]>0)),(self.physical,((intent==1)|(intent==2))&(x[:,17]>0))]:
   if mask.any():
    h=module.encoder(torch.cat([module.task(tid[mask]),x[mask,16:17]],-1));h=h+module.temporal(ft[mask]);r[mask]+=module.project(h)
  mask=x[:,25]>0
  if mask.any():
   z=x[mask];ph=phase[mask];v=torch.stack([z[:,22],z[:,23],z[:,24],ph,ph.sin(),(2*math.pi*ph).sin(),(2*math.pi*ph).cos(),(1-ph)*z[:,22],ph*0+1],-1);r[mask]+=self.goal.project(self.goal.encoder(v))
  mask=(intent==2)&(x[:,27]>0)
  if mask.any():
   ph=phase[mask];v=[cmd[mask],x[mask,26]*.03+.84,ph]
   for k in [1,2,3]:v.extend([(2*math.pi*k*ph).sin(),(2*math.pi*k*ph).cos()])
   v.append(ph*0+1);v=torch.stack(v,-1);r[mask]+=self.reach.project(self.reach.encoder(v))
  mask=intent==3
  if mask.any():
   ph=phase[mask];back=tid[mask]==6;cc=cmd[mask];v=[back.to(x),(~back).to(x),torch.where(back,(cc-.35)/.07,(cc-.55)/.15),ph,ph*0+1]
   for k in [1,3,5]:v.extend([(2*math.pi*k*ph).sin(),(2*math.pi*k*ph).cos()])
   v=torch.stack(v,-1);r[mask]+=self.exit.project(self.exit.encoder(v));o[mask]+=self.exit.output_head(v)
  return r.reshape(*shape,4,512),o.reshape(*shape,205)
