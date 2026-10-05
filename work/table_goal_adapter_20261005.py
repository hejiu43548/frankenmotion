"""Learned endpoint distance/direction branch, frozen FrankenMotion backbone."""
import os,sys,json,time,math,argparse,random,hashlib
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/table_demo_20261005';sys.path.insert(0,str(R/'work'))
import physical_adapter_20261003 as pa
from core import torch,nn,np,FK,sample
from generate import inputs
BASE=R/'outputs_amass/franken_improve_20261003/frozen_generation_v1/physical.pt'
class GoalControl(nn.Module):
 def __init__(self,base):
  super().__init__();self.base=base
  for p in base.parameters():p.requires_grad_(False)
  dim=base.root.base.latent_dim
  self.encoder=nn.Sequential(nn.Linear(9,128),nn.SiLU(),nn.Linear(128,dim))
  self.residuals=nn.ModuleList([nn.Sequential(nn.Linear(dim,128),nn.SiLU(),nn.Linear(128,dim)) for _ in base.root.base.seqTransEncoder.layers])
  for x in self.residuals:nn.init.zeros_(x[-1].weight);nn.init.zeros_(x[-1].bias)
  self.goal=None;self.features=None
  self.handles=[layer.register_forward_hook(self.hook(i)) for i,layer in enumerate(base.root.base.seqTransEncoder.layers)]
 def hook(self,i):
  def f(module,args,out):
   if self.features is None:return out
   residual=self.residuals[i](self.features);return out+torch.nn.functional.pad(residual,(0,0,out.shape[1]-residual.shape[1],0))
  return f
 def forward(self,x,y,t,tf=None):
  if self.goal is None:self.features=None
  else:
   dist,angle=self.goal.unbind(-1);phase=torch.linspace(0,1,x.shape[1],device=x.device)[None].expand(x.shape[0],-1)
   v=[dist[:,None].expand_as(phase)/3,angle[:,None].sin().expand_as(phase),angle[:,None].cos().expand_as(phase),phase,phase.sin(),torch.sin(phase*2*math.pi),torch.cos(phase*2*math.pi),(1-phase)*dist[:,None]/3,phase*0+1]
   self.features=self.encoder(torch.stack(v,-1))
  try:return self.base(x,y,t,tf)
  finally:self.features=None
 def adapter_state(self):return {k:v for k,v in self.state_dict().items() if not k.startswith('base.')}
 def train(self,mode=True):super().train(mode);self.base.eval();return self

def load(weight=None):
 m,_=pa.core.load_model(task_weights=BASE);m.denoiser=GoalControl(m.denoiser).cuda()
 if weight:
  s=torch.load(weight,map_location='cpu',weights_only=False);r=m.denoiser.load_state_dict(s['adapter'],strict=False);assert not r.unexpected_keys and all(x.startswith('base.') for x in r.missing_keys)
 return m.eval()
def setup(src,goals):
 goals=torch.as_tensor(goals,dtype=torch.float32,device='cuda');dist,ang=goals.unbind(-1);speed=dist/4.3*pa.HH/FK().height
 local,tx,cmd,_=inputs(src,speed.tolist());n=local.shape[1];tt=torch.arange(n,device='cuda')*.05
 profile=torch.clamp((tt-.6)/.6,0,1)*torch.clamp((5.3-tt)/.8,0,1);profile=profile/(profile[:-1].sum()*.05)
 values=torch.zeros(len(goals),n,2,device='cuda');values[:,:,0]=dist[:,None]*profile
 yawrate=(tt<1.2).float()/1.2;values[:,:,1]=ang[:,None]*yawrate
 controls=pa.core.encode_control(values,torch.ones(len(goals),n,device='cuda',dtype=torch.bool))
 return local,tx,cmd,controls,goals

def metrics(raw,fk,goals):
 p=fk(raw);root=p[:,:,0];d=goals[:,0];ang=goals[:,1];target=torch.stack([d*ang.cos(),d*ang.sin()],-1);xy=root[:,:,:2]-root[:,:1,:2]
 t=torch.linspace(0,5.95,raw.shape[1],device=raw.device);u=((t-.8)/4.5).clamp(0,1);u=u*u*(3-2*u)
 path=target[:,None]*u[None,:,None];end=((xy[:,-1]-target)**2).sum(-1);trajectory=((xy-path)**2).mean()
 side=p[:,:,1,:2]-p[:,:,2,:2];yaw=torch.atan2(side[...,1],side[...,0])-math.pi/2;dyaw=torch.atan2(torch.sin(yaw-yaw[:,:1]),torch.cos(yaw-yaw[:,:1]));desired=ang[:,None]*((t/1.2).clamp(0,1)[None]);heading=(1-torch.cos(dyaw-desired)).mean()
 stop=(xy[:,-12:]-xy[:,-12:-11]).square().mean();feet=p[:,:,[7,8]];vel=(feet[:,1:]-feet[:,:-1])*20;low=feet[:,1:,:,2].detach().argmin(-1);support=vel.gather(2,low[:,:,None,None].expand(-1,-1,1,3)).squeeze(2);slip=support[:,:,:2].square().mean();height=(root[:,:,2]-root[:,:1,2]).square().mean()
 return dict(endpoint=end.mean(),path=trajectory,heading=heading,stop=stop,slip=slip,height=height),dict(endpoint_error_m=end.sqrt().detach().tolist(),xy=xy[:,-1].detach().tolist(),heading_rad=dyaw[:,-1].detach().tolist())

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=1800);ap.add_argument('--name',default='goal_v1');ap.add_argument('--initial');ap.add_argument('--denoise-steps',type=int,default=10);ap.add_argument('--lr',type=float,default=3e-5);a=ap.parse_args();folder=D/a.name;folder.mkdir(exist_ok=False);torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.14);torch.manual_seed(51005);random.seed(51005)
 m=load(a.initial);fk=FK('cuda');sources=[r for r in json.loads((pa.BASE/'evaluation_manifest.json').read_text()) if r['task']=='walk' and r['source'].endswith('_s0')];opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=a.lr);start=time.monotonic();best=1e9
 protocol=dict(goal='human skeleton metres distance and signed relative direction radians',distance_range=[.8,2.8],angle_range=[-.65,.65],frames=120,base=str(BASE),base_sha256=hashlib.sha256(BASE.read_bytes()).hexdigest(),training_seeds='51005000+step',development_seeds='52005000+i',steps=a.steps,initial=a.initial,denoise_steps=a.denoise_steps,lr=a.lr,output_editing=False)
 (folder/'protocol.json').write_text(json.dumps(protocol,indent=2))
 for step in range(a.steps+1):
  if step%150==0 or step==a.steps:
   records=[]
   with torch.no_grad():
    for k,src in enumerate(sources[:2]):
     goal=[[d,ang] for d in [1.,1.8,2.6] for ang in [-.55,0,.55]];local,tx,cmd,controls,g=setup(src,goal);m.denoiser.goal=g;raw=sample(m,local,tx,10,cmd,[52005000+k]*len(goal),True,steps=50,root_controls=controls);loss,detail=metrics(raw,fk,g);records.append(dict(prompt=k,goals=goal,**detail))
   score=float(np.mean([x for r in records for x in r['endpoint_error_m']]));state=dict(adapter=m.denoiser.adapter_state(),step=step,score=score,kind='goal_endpoint_v1');torch.save(state,folder/'last.pt')
   if score<best:best=score;torch.save(state,folder/'best.pt')
   (folder/f'validation_{step:05d}.json').write_text(json.dumps(records,indent=2));(folder/'status.json').write_text(json.dumps(dict(step=step,total=a.steps,endpoint_error_m=score,best=best,elapsed_s=time.monotonic()-start,complete=step==a.steps)));print('validation',step,score,best,flush=True)
   if step==a.steps:break
  src=random.choice(sources);goal=[[random.uniform(.8,2.8),random.uniform(-.65,.65)] for _ in range(2)];local,tx,cmd,controls,g=setup(src,goal);seeds=[51005000+step]*2
  with torch.no_grad():m.denoiser.goal=None;teacher=sample(m,local,tx,10,cmd,seeds,True,steps=10,root_controls=controls)
  m.denoiser.goal=g;opt.zero_grad(set_to_none=True);raw=sample.__wrapped__(m,local,tx,10,cmd,seeds,True,steps=a.denoise_steps,root_controls=controls);l,_=metrics(raw,fk,g)
  pose=(raw[...,10:136]-teacher[...,10:136]).square().mean();loss=8*l['endpoint']+4*l['path']+2*l['heading']+15*l['stop']+.15*l['slip']+3*l['height']+.3*pose
  assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1.);opt.step()
  if step%25==0:print(json.dumps(dict(step=step,loss=float(loss),**{k:float(v) for k,v in l.items()})),flush=True)
if __name__=='__main__':main()
