"""Learned backward/side command branch preserving the base generator's stepping style."""
import sys,json,math,time,random,argparse
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/reach_demo_20261005';sys.path.insert(0,str(R/'work'));import reach_adapter_v6_20261005 as ra
from core import torch,nn,np,FK,sample,HH,quantity
from generate import inputs
class ExitControl(nn.Module):
 def __init__(self,base):
  super().__init__();self.base=base
  for p in base.parameters():p.requires_grad_(False)
  trunk=base.base.base.root.base;dim=trunk.latent_dim;self.encoder=nn.Sequential(nn.Linear(11,128),nn.SiLU(),nn.Linear(128,dim));self.residuals=nn.ModuleList([nn.Sequential(nn.Linear(dim,128),nn.SiLU(),nn.Linear(128,dim)) for _ in trunk.seqTransEncoder.layers]);self.condition=None;self.features=None
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
   task,cmd=self.condition.unbind(-1);ph=torch.linspace(0,1,x.shape[1],device=x.device)[None].expand(x.shape[0],-1);v=[(task==0)[:,None].expand_as(ph),(task==1)[:,None].expand_as(ph),cmd[:,None].expand_as(ph),ph,ph.sin()*0+1]
   for k in [1,2,3]:v.extend([torch.sin(2*math.pi*k*ph),torch.cos(2*math.pi*k*ph)])
   self.features=self.encoder(torch.stack(v,-1))
  try:return self.base(x,y,t,tf)
  finally:self.features=None
 def adapter_state(self):return {k:v for k,v in self.state_dict().items() if not k.startswith('base.')}
 def train(self,mode=True):super().train(mode);self.base.eval();return self

def load(weight=None):
 m=ra.load(D/'reach_v6/model_00750.pt');m.denoiser=ExitControl(m.denoiser).cuda()
 if weight:
  ck=torch.load(weight,map_location='cpu',weights_only=False);s=m.denoiser.load_state_dict(ck['adapter'],strict=False);assert not s.unexpected_keys and all(k.startswith('base.') for k in s.missing_keys)
 return m.eval()
SOURCES=None
def setup(task,commands):
 global SOURCES
 if SOURCES is None:SOURCES=json.loads((ra.ga.pa.BASE/'evaluation_manifest.json').read_text())
 pi=0 if task=='back_walk' else 1;src=next(r for r in SOURCES if r['source']==f'{task}_p{pi}_s0');local,tx,cmd,_=inputs(src,commands);cond=torch.stack([cmd*0+(0 if task=='back_walk' else 1),cmd],-1);return local,tx,cmd,cond,src['task_id']

def losses(raw,teacher,fk,task,cmd):
 p=fk(raw);p0=fk(teacher).detach();axis=0 if task=='back_walk' else 1;duration=(raw.shape[1]-1)/20;desired=cmd*fk.height/HH*(duration if task=='back_walk' else 1.);delta=p0[:,-1,0,axis]-p0[:,0,0,axis];ratio=(desired/(-delta).clamp_min(.15)).clamp(.5,2.0);root=p0[:,:,0].clone();root[:,:,axis]=root[:,:1,axis]+(root[:,:,axis]-root[:,:1,axis])*ratio[:,None];relative=p0-p0[:,:,:1];target=relative+root[:,:,None];feet=[7,8,10,11];legs=[4,5,7,8,10,11]
 for j in legs:target[:,:,j,axis]=root[:,:,axis]+relative[:,:,j,axis]*ratio[:,None]
 floor=p0[:,:,[7,8],2].amin(1);lift=(p0[:,:,[7,8],2]-floor[:,None]).clamp_min(0);target[:,:,[7,8],2]=floor[:,None]+lift*(1.15 if task=='back_walk' else 1.5)
 target[:,:,[10,11],2]+=lift*(.15 if task=='back_walk' else .5)
 command=((p[:,-1,0,axis]-p[:,0,0,axis]+desired).square()).mean();trajectory=(p[:,:,0]-root).square().mean();foot=(p[:,:,feet]-target[:,:,feet]).square().mean();body=((p-p[:,:,:1])-(p0-p0[:,:,:1])).square().mean();pose=(raw[...,10:136]-teacher[...,10:136]).square().mean();height=(p[:,:,0,2]-p0[:,:,0,2]).square().mean();q=quantity(p,task,HH/fk.height);ankles=p[:,:,[7,8],2];spread=ankles.quantile(.95,dim=1)-ankles.quantile(.05,dim=1)
 return dict(command=command,trajectory=trajectory,foot=foot,body=body,pose=pose,height=height),dict(actual=q.detach().tolist(),ankle_height_range=spread.detach().tolist())

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--name',default='exit_v1');ap.add_argument('--steps',type=int,default=1800);ap.add_argument('--initial');ap.add_argument('--lr',type=float,default=2e-5);a=ap.parse_args();out=D/a.name;out.mkdir(exist_ok=False);torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.15);torch.manual_seed(89105);random.seed(89105);m=load(a.initial);fk=FK('cuda');opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=a.lr);start=time.time();best=1e9;(out/'protocol.json').write_text(json.dumps(dict(training='Base FrankenMotion text-only backward/side teacher with same noise. Learned command residual; body-pose preservation and FK foot/root losses. No previous collapsed TaskControl or RootControl input for these two skills.',targets='Training-only root and stride scaling from text-only teacher, with 15% backward / 50% lateral ankle-lift increase. No spatial pose edits at inference.',range_back_human_m_s=[.28,.42],range_side_human_m=[.4,.7],train_seed_start=89105000,dev_seed_start=89205000,steps=a.steps,denoise_steps_train=20,denoise_steps_eval=50),indent=2))
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
  m.denoiser.condition=cond;raw=sample.__wrapped__(m,local,tx,tid,cmd,seeds,False,steps=20);l,detail=losses(raw,teacher,fk,task,cmd);loss=15*l['command']+10*l['trajectory']+80*l['foot']+20*l['body']+2*l['pose']+10*l['height'];opt.zero_grad();assert torch.isfinite(loss);loss.backward();torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1.);opt.step()
  if step%50==0:print(step,task,float(loss),detail,flush=True)
if __name__=='__main__':main()
