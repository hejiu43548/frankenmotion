"""Isolated temporal task adapter and differentiable contact/semantic constraints."""
import sys,os,json,math,time,random,argparse
from pathlib import Path
BASE=Path('/home/pku/frankenmotion/outputs_amass/franken_eleven_20261003');OUT=BASE.parent/'franken_improve_20261003'
sys.path.insert(0,str(BASE/'code'));os.environ['ELEVEN_OUT']=str(BASE)
import core
from core import torch,nn,np,TASKS,RANGES,HH,FK,sample
from generate import inputs
class TemporalTaskControl(core.TaskControl):
 def __init__(self,root):
  super().__init__(root);self.temporal=nn.Linear(17,root.base.latent_dim);nn.init.zeros_(self.temporal.weight);nn.init.zeros_(self.temporal.bias)
 def hook(self,i):
  def f(module,args,output):
   if self.condition is None:return output
   features,frames=self.condition;t=torch.linspace(0,1,frames,device=output.device);freq=t.new_tensor([1,2,3,4,6,8,10,12]);ph=t[:,None]*freq[None,:]*2*np.pi
   ft=torch.cat([t[:,None],ph.sin(),ph.cos()],-1)
   res=self.residuals[i](features[:,None]+self.temporal(ft)[None]);return output+torch.nn.functional.pad(res,(0,0,output.shape[1]-frames,0))
  return f
 def load_adapter(self,s):
  r=self.load_state_dict(s,strict=False);assert not r.unexpected_keys;assert all(k.startswith(('root.','temporal.')) for k in r.missing_keys)
core.TaskControl=TemporalTaskControl

def losses(p,tid,cmd):
 n=p.shape[1];t=torch.arange(n,device=p.device)/20;feet=p[:,:,[7,8]];v=(feet[:,1:]-feet[:,:-1])*20
 root=p[:,:,0];floor=feet[:,0,:,2].amin(1).detach();zero=p.sum()*0
 # Non-jump support: low ankle chooses a support candidate, no teacher-velocity filter.
 if tid!=8:
  low=feet[:,1:,:,2].detach().argmin(-1);sv=v.gather(2,low[:,:,None,None].expand(-1,-1,1,3)).squeeze(2)
  slip=sv[...,:2].square().mean();ground=(feet[:,:,2 if False else 0,2]*0).mean()*0
  minz=feet[:,:,:,2].amin(-1);ground=((minz-floor[:,None]).square()).mean()
 else:slip=zero;ground=zero
 stationary=tid in [0,1,2,3,7,8,9]
 drift=(root[:,:,:2]-root[:,:1,:2]).square().mean() if stationary else zero
 semantic=zero
 if tid==3:
  a=p[:,16:101];lat=a[:,:,16]-a[:,:,17];lat=lat/lat.norm(dim=-1,keepdim=True).clamp_min(1e-7);signal=((a[:,:,21]-(a[:,:,16]+a[:,:,17])/2)*lat).sum(-1)*HH/1.372592926
  signal=signal-signal.mean(1,keepdim=True);target=cmd[:,None]*torch.sin(2*np.pi*3*torch.linspace(0,1,85,device=p.device))[None]
  semantic=((signal-target)/.14).square().mean()
 elif tid==9:
  vtor=(p[:,:,16]+p[:,:,17])/2-root;pitch=torch.atan2(vtor[:,:,0],vtor[:,:,2]);ramp=((t-.3)/.8).clamp(0,1);target=cmd[:,None]*ramp[None]
  semantic=((pitch-target)/.5).square().mean()
 elif tid in [6,10]:
  direction=-1 if tid==6 else 1;net=(root[:,-1,0]-root[:,0,0])/((n-1)/20)*HH/1.372592926
  semantic=((direction*net-cmd)/.55).square().mean()+((root[:,:,1]-root[:,:1,1]).square()).mean()*4
 elif tid==8:
  # Aerial part must resemble free flight, and begin/end grounded. No trajectory editing.
  height=(root[:,:,2]-root[:,:1,2])*HH/1.372592926
  airborne=(feet[:,:,2 if False else 0,2]>floor[:,None]+.08)&(feet[:,:,1,2]>floor[:,None]+.08)
  az=(root[:,2:,2]-2*root[:,1:-1,2]+root[:,:-2,2])*400;mask=airborne[:,1:-1].detach()
  ballistic=(((az+9.81)/9.81).square()*mask).sum()/mask.sum().clamp_min(1)
  landing=height[:,-1].square().mean()*10
  semantic=ballistic+landing
 return dict(slip=slip,ground=ground,drift=drift,semantic=semantic)

def evaluate(m,fk,sources,step,folder):
 rows=[]
 with torch.no_grad():
  for tid in range(11):
   src=sources[tid][0];lo,hi=RANGES[tid];cs=[lo,(lo+hi)/2,hi];local,tx,cmd,controls=inputs(src,cs)
   raw=sample(m,local,tx,tid,cmd,[990700+tid]*3,True,root_controls=controls);p=fk(raw);q=core.quantity(p,tid,HH/fk.height);l=losses(p,tid,cmd)
   for c,v in zip(cs,q.tolist()):rows.append(dict(task=TASKS[tid],command=c,quantity=v,error=abs(v-c)/(hi-lo),**{k:float(vv) for k,vv in l.items()}))
 score=float(np.mean([r['error']+r['slip']+r['semantic']*.1+2*r['drift'] for r in rows]));(folder/f'validation_{step:05d}.json').write_text(json.dumps(rows,indent=2));return score

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--steps',type=int,default=6600);a=ap.parse_args();folder=OUT/'physical_adapter';folder.mkdir(parents=True,exist_ok=True)
 torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.15);torch.manual_seed(82004);random.seed(82004)
 m,_=core.load_model(task_weights=BASE/'task_adapter_deploy.pt');fk=FK('cuda');opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=2e-5)
 manifest=json.loads((BASE/'evaluation_manifest.json').read_text());sources={i:[r for r in manifest if r['task_id']==i and r['source'].endswith('_s0')] for i in range(11)};best=float('inf');start=time.monotonic()
 (folder/'protocol.json').write_text(json.dumps(dict(steps=a.steps,train_seeds='4000000+step',development_seeds='990700+task',confirmation_seeds='reserved 42026000+prompt*100+seed',variant='learned temporal conditioning; contact and semantic losses; no output editing',warning='support-foot heuristic and ballistic root acceleration are proxies, not a proof of dynamic feasibility'),indent=2))
 for step in range(a.steps+1):
  if step%550==0 or step==a.steps:
   score=evaluate(m,fk,sources,step,folder);state=dict(adapter=m.denoiser.adapter_state(),step=step,score=score,kind='temporal_physics_v1',optimizer=opt.state_dict())
   torch.save(state,folder/'last.pt')
   if score<best:best=score;torch.save(state,folder/'best.pt')
   (folder/'status.json').write_text(json.dumps(dict(step=step,total=a.steps,score=score,best=best,elapsed_s=time.monotonic()-start,complete=step==a.steps)));print('validation',step,score,best,flush=True)
   if step==a.steps:break
  tid=step%11;src=random.choice(sources[tid]);lo,hi=RANGES[tid];cmdv=random.uniform(lo,hi);local,tx,cmd,controls=inputs(src,[cmdv]);seed=[4000000+step]
  with torch.no_grad():teacher=sample(m,local,tx,tid,cmd,seed,False,steps=10,root_controls=controls);p0=fk(teacher)
  opt.zero_grad(set_to_none=True);raw=sample.__wrapped__(m,local,tx,tid,cmd,seed,True,steps=10,root_controls=controls);p=fk(raw);q=core.quantity(p,tid,HH/fk.height)
  command=torch.nn.functional.smooth_l1_loss((q-cmd)/(hi-lo),torch.zeros_like(cmd));pose=(((p-p[:,:,:1])-(p0-p0[:,:,:1])).square()).mean();l=losses(p,tid,cmd)
  loss=command*3+pose*2+l['slip']*2+l['ground']*10+l['drift']*10+l['semantic']*.3
  if not torch.isfinite(loss):raise FloatingPointError('loss')
  loss.backward();torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1.);opt.step()
  if step%55==0:print(json.dumps(dict(step=step,task=TASKS[tid],loss=float(loss),command=float(command),**{k:float(v) for k,v in l.items()})),flush=True)
if __name__=='__main__':main()
