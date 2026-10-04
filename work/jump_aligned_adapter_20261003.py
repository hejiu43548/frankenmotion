"""Jump-only generator adapter: supervised mocap pose/velocity, no inference edits."""
import sys,json,time,random
from pathlib import Path
sys.path.insert(0,'/home/pku/frankenmotion/work')
import physical_adapter_20261003 as pa
from core import torch,np,FK,sample
from generate import inputs
BASE=pa.BASE;OUT=pa.OUT;folder=OUT/'jump_aligned_adapter';folder.mkdir(exist_ok=True)
torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.12);torch.manual_seed(7104);random.seed(7104)
m,_=pa.core.load_model(task_weights=OUT/'physical_adapter/best.pt');fk=FK('cuda');opt=torch.optim.AdamW([p for p in m.parameters() if p.requires_grad],lr=2e-5)
keys=['03371','04366','03664','02982','05586'];dataset=json.loads((BASE/'data_manifest.json').read_text());examples=[]
for key in keys:
 r=next(r for r in dataset if r['task']=='jump' and r['split']=='train' and r['key']==key);raw=torch.load(r['path'],map_location='cuda',weights_only=False)['x'][None,:,:205]
 with torch.no_grad():
  p=fk(raw);peak=int(p[0,:,0,2].argmax());ids=(torch.arange(60,device='cuda')+peak-30).clamp(0,59);p=p[:,ids].clone();p[:,:,:,:2]-=p[:,:1,:1,:2].clone();q=float(pa.core.quantity(p,8,pa.HH/fk.height)[0])
 examples.append((key,p,q))
sources=[r for r in json.loads((BASE/'evaluation_manifest.json').read_text()) if r['task']=='jump' and r['source'].endswith('_s0')]
(folder/'protocol.json').write_text(json.dumps(dict(teacher_keys=keys,teacher_commands=[x[2] for x in examples],teacher_selection='TRAIN mocap passing GMR+SONIC, one flight per clip; integer-time translation aligns root peak to frame30, holds endpoints; no time rescaling, no inference edits; no heldout confirmation',train_noise='7104000+step',development_noise=990708,steps=4400,loss='full canonical FK pose, leg FK pose, joint-point velocity, command',inference='50-step DDIM, jump-only checkpoint selection; no pose editing',limitation='Highest teacher 0.51m; 0.55m is extrapolation'),indent=2))
def validate(step):
 cs=[.25,.325,.4,.475,.55];local,tx,cmd,controls=inputs(sources[0],cs)
 raw=sample(m,local,tx,8,cmd,[990708]*5,True,root_controls=controls)
 with torch.no_grad():p=fk(raw);q=pa.core.quantity(p,8,pa.HH/fk.height);err=[]
 for i,c in enumerate(cs):
  key,target,tq=min(examples,key=lambda e:abs(e[2]-c));err.append(float((p[i:i+1]-target).square().mean()))
 score=float(((q-cmd).abs()/.3).mean())+10*np.mean(err);state=dict(adapter=m.denoiser.adapter_state(),step=step,score=score,kind='jump_aligned_imitation_v1');torch.save(state,folder/'last.pt')
 (folder/f'validation_{step:05d}.json').write_text(json.dumps(dict(commands=cs,quantity=q.tolist(),pose_mse=err,score=score)))
 print('validation',step,q.tolist(),score,flush=True);return score,state
best=float('inf');start=time.monotonic()
for step in range(4401):
 if step%550==0:
  score,state=validate(step)
  if score<best:best=score;torch.save(state,folder/'best.pt')
  (folder/'status.json').write_text(json.dumps(dict(step=step,best=best,elapsed_s=time.monotonic()-start,complete=step==4400)))
  if step==4400:break
 key,target,c=random.choice(examples);src=random.choice(sources);local,tx,cmd,controls=inputs(src,[c]);opt.zero_grad(set_to_none=True)
 raw=sample.__wrapped__(m,local,tx,8,cmd,[7104000+step],True,steps=10,root_controls=controls);p=fk(raw);q=pa.core.quantity(p,8,pa.HH/fk.height)
 pose=(p-target).square().mean();legs=(p[:,:,[1,2,4,5,7,8]]-target[:,:,[1,2,4,5,7,8]]).square().mean();vel=((p[:,1:]-p[:,:-1]-target[:,1:]+target[:,:-1])*20).square().mean();command=((q-c)/.3).square().mean();loss=pose*15+legs*25+vel*.3+command*2
 if not torch.isfinite(loss):raise FloatingPointError('loss')
 loss.backward();torch.nn.utils.clip_grad_norm_([p for p in m.parameters() if p.requires_grad],1);opt.step()
 if step%110==0:print(step,key,'loss',float(loss),'pose',float(pose),'vel',float(vel),flush=True)
