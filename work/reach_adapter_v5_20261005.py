"""Learned reach/place/retract command branch; no output-space IK or hand-path editing."""
import sys,json,time,math,argparse,random,hashlib
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/reach_demo_20261005';sys.path.insert(0,str(R/'work'))
import table_goal_adapter_20261005 as ga
from core import torch,nn,np,FK,sample,HH,rotation_6d_to_matrix,axis_angle_rotation,matrix_to_euler_angles
from generate import inputs
RH=1.0486437524221748
class ReachControl(nn.Module):
 def __init__(self,base):
  super().__init__();self.base=base
  for p in base.parameters():p.requires_grad_(False)
  trunk=base.base.root.base;dim=trunk.latent_dim;self.encoder=nn.Sequential(nn.Linear(10,128),nn.SiLU(),nn.Linear(128,dim));self.residuals=nn.ModuleList([nn.Sequential(nn.Linear(dim,128),nn.SiLU(),nn.Linear(128,dim)) for _ in trunk.seqTransEncoder.layers]);self.command=None;self.features=None
  for x in self.residuals:nn.init.zeros_(x[-1].weight);nn.init.zeros_(x[-1].bias)
  self.handles=[layer.register_forward_hook(self.hook(i)) for i,layer in enumerate(trunk.seqTransEncoder.layers)]
 def hook(self,i):
  def f(module,args,out):
   if self.features is None:return out
   res=self.residuals[i](self.features);return out+torch.nn.functional.pad(res,(0,0,out.shape[1]-res.shape[1],0))
  return f
 def forward(self,x,y,t,tf=None):
  if self.command is None:self.features=None
  else:
   cmd,height=self.command.unbind(-1);ph=torch.linspace(0,1,x.shape[1],device=x.device)[None].expand(x.shape[0],-1);v=[cmd[:,None].expand_as(ph),height[:,None].expand_as(ph),ph,torch.sin(2*math.pi*ph),torch.cos(2*math.pi*ph),torch.sin(4*math.pi*ph),torch.cos(4*math.pi*ph),torch.sin(6*math.pi*ph),torch.cos(6*math.pi*ph),ph*0+1];self.features=self.encoder(torch.stack(v,-1))
  try:return self.base(x,y,t,tf)
  finally:self.features=None
 def adapter_state(self):return {k:v for k,v in self.state_dict().items() if not k.startswith('base.')}
 def train(self,mode=True):super().train(mode);self.base.eval();return self

def load(weight=None):
 m=ga.load(R/'outputs_amass/gait_demo_20261005/frozen/goal_adapter.pt');m.denoiser=ReachControl(m.denoiser).cuda()
 if weight:
  s=torch.load(weight,map_location='cpu',weights_only=False);r=m.denoiser.load_state_dict(s['adapter'],strict=False);assert not r.unexpected_keys and all(x.startswith('base.') for x in r.missing_keys)
 return m.eval()
def setup(commands,heights=None,prompt=0):
 heights=heights or [.84]*len(commands);src=dict(prompt=str(D/f'prompts/reach_place_p{prompt}.json'),task='reach',frames=120);local,tx,cmd,controls=inputs(src,commands);cond=torch.tensor(list(zip(commands,heights)),device='cuda',dtype=torch.float32);return local,tx,cmd,controls,cond

def targets(cmd,height,fk,device,n=120):
 t=torch.arange(n,device=device)*.05;scale=RH/fk.height;b=len(cmd);low=torch.stack([cmd*0+.04,cmd*0-.23,cmd*0+.64],-1);lift=torch.stack([cmd*0+.00,cmd*0-.18,height+.14],-1);above=torch.stack([cmd*RH/HH,cmd*0-.18,height+.14],-1);held=above.clone();held[:,2]=height-(.035+.12*(cmd-.3))
 values=[low,lift,above,held,held,above,lift,low];times=[0,.8,1.8,2.4,3.6,4.2,5.2,5.95];target=low[:,None].expand(b,n,3).clone()
 for i in range(len(times)-1):
  u=((t-times[i])/(times[i+1]-times[i])).clamp(0,1);u=u*u*(3-2*u);mask=(t>=times[i])&(t<=times[i+1]);target[:,mask]=values[i][:,None]+u[mask][None,:,None]*(values[i+1]-values[i])[:,None]
 return target/scale,t

def losses(raw,fk,cond,teacher):
 p=fk(raw);pt=fk(teacher);b,n=p.shape[:2];target,t=targets(cond[:,0],cond[:,1],fk,raw.device,n);floor=pt[:,:1,[7,8],2].amin(-1).detach();target[:,:,2]+=floor-.035/(RH/fk.height);target[:,:,:2]+=pt[:,:1,0,:2];hand=(p[:,:,21]-target).square().mean();holdmask=(t>=2.45)&(t<=3.55);hold=(p[:,holdmask,21]-target[:,holdmask]).square().mean();bodyids=[0,1,2,3,4,5,6,7,8,9,10,11,12,13,15,16,18,20];body=(p[:,:,bodyids]-pt[:,:1,bodyids]).square().mean();vel=(p[:,1:,21]-p[:,:-1,21])*20;acc=(vel[:,1:]-vel[:,:-1])*20;smooth=acc.square().mean();root=(p[:,:,0]-pt[:,:1,0]).square().mean();feet=(p[:,:,[7,8,10,11]]-pt[:,:1,[7,8,10,11]]).square().mean()
 mat=rotation_6d_to_matrix(raw[...,4:136].reshape(b,n,22,6));e=matrix_to_euler_angles(mat[:,:,0],'ZYX');yaw=torch.cat([torch.zeros_like(raw[:,:1,3]),torch.cumsum(raw[:,:-1,3],1)],1);m0=axis_angle_rotation('Z',yaw)@axis_angle_rotation('Y',e[...,1])@axis_angle_rotation('X',e[...,2]);g=[]
 for j in range(22):g.append(m0 if j==0 else g[fk.parents[j]]@mat[:,:,j])
 uncanonical=fk(raw,canonical=False);side=uncanonical[:,0,1,:2]-uncanonical[:,0,2,:2];initial_yaw=torch.atan2(side[:,1],side[:,0])-math.pi/2;hand_canonical=axis_angle_rotation('Z',-initial_yaw)[:,None]@g[21];desired=torch.diag(raw.new_tensor([-1.,-1.,1.]));orient=(hand_canonical[:,(t>.8)&(t<5.2)]-desired).square().mean();pose=(raw[...,10:136]-teacher[...,10:136]).square().mean();return dict(hand=hand,hold=hold,body=body,smooth=smooth,root=root,feet=feet,orient=orient,pose=pose),dict(hand_rmse_robot_m=float((hand*3).sqrt().detach())*RH/fk.height,hold_rmse_robot_m=float((hold*3).sqrt().detach())*RH/fk.height)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--name',default='reach_v5');ap.add_argument('--steps',type=int,default=1800);ap.add_argument('--initial');ap.add_argument('--denoise-steps',type=int,default=10);ap.add_argument('--lr',type=float,default=3e-5);a=ap.parse_args();out=D/a.name;out.mkdir(exist_ok=False);torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.16);torch.manual_seed(81005);random.seed(81005);m=load(a.initial);fk=FK('cuda');opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=a.lr);best=1e9;start=time.time();(out/'protocol.json').write_text(json.dumps(dict(kind='learned reach temporal branch with differentiable human FK target losses',range_command_human_equiv_m=[.28,.52],range_palm_height_robot_m=[.81,.87],steps=a.steps,inference_steps=50,training_steps=a.denoise_steps,hand_trajectory='Training targets only; held wrist height includes fixed command-dependent height calibration .035+.12*(command-.3) metres measured on development GMR results. Retract/lift wrist target x=0.00m provides clearance before lowering. No pose edits or scene IK at inference',command_definition='right wrist forward distance from pelvis, human-equivalent metres; G1 palm has additional geometric offset',base_goal=str(R/'outputs_amass/gait_demo_20261005/frozen/goal_adapter.pt'),train_seed_start=82005000,validation_seed_start=83005000),indent=2))
 for step in range(a.steps+1):
  if step%150==0 or step==a.steps:
   local,tx,cmd,controls,cond=setup([.3,.4,.5]);m.denoiser.command=None
   with torch.no_grad():teacher=sample(m,local,tx,1,cmd,[83005000]*3,True,steps=50);m.denoiser.command=cond;raw=sample(m,local,tx,1,cmd,[83005000]*3,True,steps=50);l,metrics=losses(raw,fk,cond,teacher)
   score=metrics['hold_rmse_robot_m']+metrics['hand_rmse_robot_m'];state=dict(adapter=m.denoiser.adapter_state(),step=step,score=score,kind='reach_place_v5_retract_clearance');torch.save(state,out/'last.pt');torch.save(state,out/f'model_{step:05d}.pt')
   if score<best:best=score;torch.save(state,out/'best.pt')
   np.savez_compressed(out/f'validation_{step:05d}.npz',motion=raw.cpu().numpy(),commands=cmd.cpu().numpy());(out/'status.json').write_text(json.dumps(dict(step=step,total=a.steps,score=score,best=best,elapsed_s=time.time()-start,complete=step==a.steps,**metrics)));print('validation',step,metrics,'score',score,flush=True)
   if step==a.steps:break
  cs=[random.uniform(.28,.52) for _ in range(2)];hs=[random.uniform(.81,.87) for _ in range(2)];local,tx,cmd,controls,cond=setup(cs,hs,step%2);seeds=[82005000+step]*2
  with torch.no_grad():m.denoiser.command=None;teacher=sample(m,local,tx,1,cmd,seeds,True,steps=10)
  m.denoiser.command=cond;opt.zero_grad(set_to_none=True);raw=sample.__wrapped__(m,local,tx,1,cmd,seeds,True,steps=a.denoise_steps);l,metrics=losses(raw,fk,cond,teacher);loss=40*l['hand']+60*l['hold']+8*l['body']+20*l['root']+20*l['feet']+.00005*l['smooth']+.3*l['orient']+.015*l['pose'];assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1.);opt.step()
  if step%25==0:print(step,float(loss),metrics,flush=True)
if __name__=='__main__':main()
