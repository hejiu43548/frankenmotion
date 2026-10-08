"""Learned backward/side command branch preserving the base generator's stepping style."""
import sys,json,math,time,random,argparse
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/reach_demo_20261005';sys.path.insert(0,str(R/'work'));import reach_adapter_v6_20261005 as ra
from core import torch,nn,np,FK,sample,HH,quantity,axis_angle_rotation,rotation_6d_to_matrix,matrix_to_euler_angles
from generate import inputs
class ExitControl(nn.Module):
 def __init__(self,base):
  super().__init__();self.base=base
  for p in base.parameters():p.requires_grad_(False)
  trunk=base.base.base.root.base;dim=trunk.latent_dim;self.encoder=nn.Sequential(nn.Linear(11,128),nn.SiLU(),nn.Linear(128,dim));self.residuals=nn.ModuleList([nn.Sequential(nn.Linear(dim,128),nn.SiLU(),nn.Linear(128,dim)) for _ in trunk.seqTransEncoder.layers]);self.condition=None;self.features=None;self.output_head=nn.Sequential(nn.Linear(11,128),nn.SiLU(),nn.Linear(128,205));nn.init.zeros_(self.output_head[-1].weight);nn.init.zeros_(self.output_head[-1].bias)
  for x in self.residuals:nn.init.zeros_(x[-1].weight);nn.init.zeros_(x[-1].bias)
  self.handles=[layer.register_forward_hook(self.hook(i)) for i,layer in enumerate(trunk.seqTransEncoder.layers)]
 def hook(self,i):
  def f(module,args,out):
   if self.features is None:return out
   z=self.residuals[i](self.features);return out+torch.nn.functional.pad(z,(0,0,out.shape[1]-z.shape[1],0))
  return f
 def forward(self,x,y,t,tf=None):
  if self.condition is None:self.features=None
  else:
   task,cmd=self.condition.unbind(-1);ph=torch.linspace(0,1,x.shape[1],device=x.device)[None].expand(x.shape[0],-1);v=[(task==0)[:,None].expand_as(ph),(task==1)[:,None].expand_as(ph),torch.where(task==0,(cmd-.35)/.07,(cmd-.55)/.15)[:,None].expand_as(ph),ph,ph.sin()*0+1]
   for k in [1,3,5]:v.extend([torch.sin(2*math.pi*k*ph),torch.cos(2*math.pi*k*ph)])
   condition_features=torch.stack(v,-1);self.features=self.encoder(condition_features)
  try:
   out=self.base(x,y,t,tf)
   if self.condition is not None:out=out+torch.nn.functional.pad(self.output_head(condition_features),(0,out.shape[-1]-205))
   return out
  finally:self.features=None
 def adapter_state(self):return {k:v for k,v in self.state_dict().items() if not k.startswith('base.')}
 def train(self,mode=True):super().train(mode);self.base.eval();return self

def load(weight=None,reach_weight=None):
 m=ra.load(reach_weight or D/'reach_v6/model_00750.pt');m.denoiser=ExitControl(m.denoiser).cuda()
 if weight:
  ck=torch.load(weight,map_location='cpu',weights_only=False);s=m.denoiser.load_state_dict(ck['adapter'],strict=False);assert not s.unexpected_keys and all(k.startswith(('base.','output_head.')) for k in s.missing_keys)
 return m.eval()
SOURCES=None
def setup(task,commands):
 global SOURCES
 if SOURCES is None:SOURCES=json.loads((ra.ga.pa.BASE/'evaluation_manifest.json').read_text())
 pi=0 if task=='back_walk' else 1;src=next(r for r in SOURCES if r['source']==f'{task}_p{pi}_s0');local,tx,cmd,_=inputs(src,commands);cond=torch.stack([cmd*0+(0 if task=='back_walk' else 1),cmd],-1);return local,tx,cmd,cond,src['task_id']

def losses(raw,teacher,fk,task,cmd):
 p=fk(raw);p0=fk(teacher).detach();b,n=p.shape[:2];device=p.device;axis=0 if task=='back_walk' else 1;T=(n-1)/20;desired=cmd*fk.height/HH*(T if task=='back_walk' else 1.);t=torch.arange(n,device=device)*.05;period=1.2 if task=='back_walk' else 2.;ramp=.5
 def progress(tt):
  tt=tt.clamp(0,T);return torch.where(tt<ramp,.5*tt.square()/ramp,torch.where(tt>T-ramp,T-ramp-.5*(T-tt).square()/ramp,tt-ramp/2))/(T-ramp)
 root=p0[:,:1,0].expand(-1,n,-1).clone();root[:,:,2]-=.025;root[:,:,axis]=root[:,:1,axis]-desired[:,None]*progress(t)[None];side=p0[:,:,1,:2]-p0[:,:,2,:2];yaw=torch.atan2(side[:,:,1],side[:,:,0])-math.pi/2;relative=(axis_angle_rotation('Z',-yaw)[:,:,None]@(p0-p0[:,:,:1])[...,None]).squeeze(-1);target=relative+root[:,:,None];floor=p0[:,:,[7,8],2].quantile(.05,dim=1).amin(-1);lift=.080 if task=='back_walk' else .070;knee_targets=[]
 for j,off,toe in [(7,0.,10),(8,.5,11)]:
  phase=t/period+off;cycle=torch.floor(phase);u=((phase-cycle-.6)/.4).clamp(0,1);knee_targets.append(.18+.65*torch.sin(math.pi*u).square());sm=u*u*(3-2*u);centre=(cycle-off+.3)*period;anchor0=-desired[:,None]*progress(centre)[None];anchor1=-desired[:,None]*progress(centre+period)[None];base=relative[:,0,j].clone();base[:,axis]=0.;want=root[:,:1]+base[:,None];want=want.expand(-1,n,-1).clone();want[:,:,axis]=root[:,:1,axis]+anchor0+(anchor1-anchor0)*sm[None];want[:,:,2]=floor[:,None]+lift*torch.sin(math.pi*u).square()[None];target[:,:,j]=want;toeoffset=relative[:,:,toe]-relative[:,:,j];toeoffset=toeoffset.mean(1);target[:,:,toe]=want+toeoffset[:,None]
 feet=[7,8,10,11];upper=[1,2,3,6,9,12,13,14,15,16,17,18,19,20,21];command=((p[:,-1,0,axis]-p[:,0,0,axis]+desired).square()).mean();trajectory=(p[:,:,0]-root).square().mean();foot=(p[:,:,feet]-target[:,:,feet]).square().mean();body=((p-p[:,:,:1])[:,:,upper]-relative[:,:,upper]).square().mean();pose=(raw[...,10:136]-teacher[...,10:136]).square().mean();height=(p[:,:,0,2]-root[:,:,2]).square().mean();q=quantity(p,task,HH/fk.height);ankles=p[:,:,[7,8],2];spread=ankles.quantile(.95,dim=1)-ankles.quantile(.05,dim=1)
 mat=rotation_6d_to_matrix(raw[...,4:136].reshape(b,n,22,6));e=matrix_to_euler_angles(mat[:,:,0],'ZYX');rawyaw=torch.cat([torch.zeros_like(raw[:,:1,3]),torch.cumsum(raw[:,:-1,3],1)],1);m0=axis_angle_rotation('Z',rawyaw)@axis_angle_rotation('Y',e[...,1])@axis_angle_rotation('X',e[...,2]);g=[]
 for j in range(22):g.append(m0 if j==0 else g[fk.parents[j]]@mat[:,:,j])
 pu=fk(raw,canonical=False);side0=pu[:,0,1,:2]-pu[:,0,2,:2];iy=torch.atan2(side0[:,1],side0[:,0])-math.pi/2;footrot=axis_angle_rotation('Z',-iy)[:,None,None]@torch.stack([g[10],g[11]],2);desiredrot=raw.new_tensor([[0.,0.,1.],[1.,0.,0.],[0.,1.,0.]]);orientation=(footrot-desiredrot).square().mean();wantknees=axis_angle_rotation('X',torch.stack(knee_targets,1));knee=(mat[:,:,[4,5]]-wantknees[None]).square().sum((-1,-2)).mean();pside=p[:,:,1,:2]-p[:,:,2,:2];heading=(1-torch.cos(torch.atan2(pside[:,:,1],pside[:,:,0])-math.pi/2)).mean()
 return dict(command=command,trajectory=trajectory,foot=foot,foot_height=(p[:,:,feet,2]-target[:,:,feet,2]).square().mean(),body=body,pose=pose,height=height,orientation=orientation,heading=heading,knee=knee),dict(actual=q.detach().tolist(),ankle_height_range=spread.detach().tolist())

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--name',default='exit_v5');ap.add_argument('--steps',type=int,default=1800);ap.add_argument('--initial');ap.add_argument('--initial-v1',action='store_true');ap.add_argument('--lr',type=float,default=2e-5);a=ap.parse_args();out=D/a.name;out.mkdir(exist_ok=False);torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.15);torch.manual_seed(89105);random.seed(89105);m=load(a.initial);
 if a.initial_v1:
  with torch.no_grad():
   w=m.denoiser.encoder[0].weight;old=w[:,2].clone();w[:,0]+=.35*old;w[:,1]+=.55*old;w[:,2]*=.1
 fk=FK('cuda');opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=a.lr);start=time.time();best=1e9;(out/'protocol.json').write_text(json.dumps(dict(architecture='Latent hooks plus learned per-frame conditional denoiser output residual; not post-sampling pose edits',command_features='Task-wise normalized continuous parameter',training='Learned command residual with training-only alternating-contact FK gait supervision and base text motion upper-body prior. No previous collapsed TaskControl or RootControl input for these two skills.',targets='Training-only smooth root travel and alternating fixed-contact/swing ankle waypoints: backward1.2s cycle/8cm human lift, side2s cycle/7cm lift; flat foot orientation, constant facing, 2.5cm lower root, and direct knee-flexion rotation supervision (0.18 to0.83rad) to avoid straight-leg FK gradient collapse. Base text motion supplies upper-body prior. Learned FK supervision only; no output-space gait synthesis or IK after inference.',range_back_human_m_s=[.28,.42],range_side_human_m=[.4,.7],train_seed_start=89105000,dev_seed_start=89205000,steps=a.steps,denoise_steps_train=20,denoise_steps_eval=50),indent=2))
 for step in range(a.steps+1):
  if step%300==0 or step==a.steps:
   rows=[]
   with torch.no_grad():
    for task,cs in [('back_walk',[.3,.35,.4]),('sidestep',[.4,.5,.65])]:
     local,tx,cmd,cond,tid=setup(task,cs);m.denoiser.condition=None;teacher=sample(m,local,tx,tid,cmd,[89205000]*3,False,steps=50);m.denoiser.condition=cond;raw=sample(m,local,tx,tid,cmd,[89205000]*3,False,steps=50);l,detail=losses(raw,teacher,fk,task,cmd);rows.append(dict(task=task,commands=cs,**detail));np.savez_compressed(out/f'validation_{step:05d}_{task}.npz',motion=raw.cpu().numpy(),commands=cs)
   score=sum(sum(abs(x-c) for x,c in zip(r['actual'],r['commands'])) for r in rows);ck=dict(adapter=m.denoiser.adapter_state(),step=step,score=score);torch.save(ck,out/f'model_{step:05d}.pt');torch.save(ck,out/'last.pt')
   if score<best:best=score;torch.save(ck,out/'best.pt')
   (out/'status.json').write_text(json.dumps(dict(step=step,elapsed_s=time.time()-start,score=score,complete=step==a.steps,rows=rows),indent=2));print('validation',step,rows,flush=True)
   if step==a.steps:break
  task='back_walk' if step%2==0 else 'sidestep';cs=[random.uniform(.28,.42) if task=='back_walk' else random.uniform(.4,.7) for _ in range(2)];local,tx,cmd,cond,tid=setup(task,cs);seeds=[89105000+step]*2
  with torch.no_grad():m.denoiser.condition=None;teacher=sample(m,local,tx,tid,cmd,seeds,False,steps=20)
  m.denoiser.condition=cond;raw=sample.__wrapped__(m,local,tx,tid,cmd,seeds,False,steps=20);l,detail=losses(raw,teacher,fk,task,cmd);loss=15*l['command']+15*l['trajectory']+160*l['foot']+160*l['foot_height']+15*l['body']+.1*l['pose']+10*l['height']+.5*l['orientation']+2*l['heading']+2*l['knee'];opt.zero_grad();assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1.);opt.step()
  if step%50==0:print(step,task,float(loss),detail,flush=True)
if __name__=='__main__':main()
