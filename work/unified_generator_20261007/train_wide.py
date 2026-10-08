"""Shared denoiser distillation: one task-conditioned adapter, training-only teachers."""
from common import *
from wide_adapter import enable
enable()
import argparse,time,random,hashlib
p=argparse.ArgumentParser();p.add_argument('--name',default='shared_v1');p.add_argument('--steps',type=int,default=11000);p.add_argument('--lr',type=float,default=3e-5);p.add_argument('--initial');p.add_argument('--recon-weight',type=float,default=.05);p.add_argument('--command-weight',type=float,default=.02);a=p.parse_args();out=D/'training'/a.name;out.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.15);torch.manual_seed(710710);random.seed(710710)
m,_=pa.core.load_model('cuda',task_weights=Path(a.initial) if a.initial else D/'backup/physical.pt');fk=FK('cuda');params=[p for p in m.parameters() if p.requires_grad];opt=torch.optim.AdamW(params,lr=a.lr);records=json.loads((D/'teacher_training/manifest.json').read_text());groups={i:[r for r in records if r['task_id']==i] for i in range(11)};cache={}
teachers={t:m.denoiser.expand_adapter(torch.load(teacher_path(t),map_location='cpu',weights_only=False)['adapter']) for t in TASKS};adapter_keys=list(m.denoiser.adapter_state());named=dict(m.denoiser.named_parameters());assert all(k in named for k in adapter_keys)
for teacher in teachers.values():
 assert not set(teacher)-set(adapter_keys)
 for k in adapter_keys:
  if k not in teacher:
   assert k.startswith('temporal.');teacher[k]=torch.zeros_like(named[k],device='cpu')
for r in records:
 z=np.load(r['path']);cache[r['path']]=torch.from_numpy(z['motion'].copy()).float()
protocol=dict(kind=__doc__,steps=a.steps,learning_rate=a.lr,seed=710710,initial=a.initial or 'backup/physical.pt',recon_weight=a.recon_weight,command_weight=a.command_weight,tasks=TASKS,teacher_policy='Historical specialist adapters used only as frozen training targets, switched temporarily for no-grad denoising on identical noisy input; restored student parameters before its forward. Export contains one adapter, no teacher bank or routing.',train_sources='prompt p0/p1,3 new noises107072000+100*prompt+noise,5commands each,330 clips. Historical880 audit/test excluded.',loss='normalized teacher denoiser prediction MSE,first4 features weight5 +0.02 command Huber/span +0.05 reconstructed clean training target MSE',batch_size=1,trainable_parameters=sum(x.numel() for x in params),teacher_hashes={t:hashlib.sha256(teacher_path(t).read_bytes()).hexdigest() for t in TASKS},scope='Existing task ID and numerical command conditions retained; Shared residual hidden width512; no experts or dispatch. Narrow initialization preserved by zero output columns for new neurons.')
(out/'protocol.json').write_text(json.dumps(protocol,indent=2));start=time.monotonic()
def save(step):
 f=out/f'model_{step:05d}.pt';tmp=f.with_suffix('.tmp');torch.save(dict(adapter={k:v.detach().cpu().clone() for k,v in m.denoiser.adapter_state().items()},step=step,kind='one_shared_wide_temporal_task_adapter',tasks=TASKS,ranges=RANGES,optimizer=opt.state_dict()),tmp);tmp.replace(f);f.with_suffix('.ready.json').write_text(json.dumps(dict(sha256=hashlib.sha256(f.read_bytes()).hexdigest())))
save(0)
for step in range(1,a.steps+1):
 tid=(step-1)%11;r=random.choice(groups[tid]);local,tx,cmd,controls=inputs(r,[r['command']],'cuda',fk);raw=cache[r['path']][None].cuda();clean=m.motion_normalizer(torch.cat([raw,local],-1));t=torch.randint(0,m.timesteps,(1,),device='cuda');noise=torch.randn_like(clean);xt=m.q_sample(xstart=clean,t=t,noise=noise);xt=torch.cat([xt[...,:205],clean[...,205:]],-1);y=pa.core.make_y(m,clean,tx,tid,cmd,True,controls)
 student={k:named[k].detach().clone() for k in adapter_keys}
 with torch.no_grad():
  for k in adapter_keys:named[k].copy_(teachers[TASKS[tid]][k])
  target=m.denoiser(xt,y,t).detach()
  for k in adapter_keys:named[k].copy_(student[k])
 opt.zero_grad(set_to_none=True);pred=m.denoiser(xt,y,t);weights=torch.ones(205,device='cuda');weights[:4]=5;distill=((pred[...,:205]-target[...,:205]).square()*weights).mean();recon=(pred[...,:205]-clean[...,:205]).square().mean();decoded=m.motion_normalizer.inverse(pred)[...,:205];q=pa.core.quantity(fk(decoded),tid,HH/fk.height);lo,hi=RANGES[tid];command=torch.nn.functional.smooth_l1_loss((q-cmd)/(hi-lo),torch.zeros_like(cmd));loss=distill+a.recon_weight*recon+a.command_weight*command
 if not torch.isfinite(loss):raise FloatingPointError('nonfinite loss')
 loss.backward();norm=torch.nn.utils.clip_grad_norm_(params,1.)
 if not torch.isfinite(norm):raise FloatingPointError('nonfinite gradient')
 opt.step()
 if step%110==0:
  status=dict(step=step,total=a.steps,task=TASKS[tid],loss=float(loss),distill=float(distill),recon=float(recon),command=float(command),elapsed_s=time.monotonic()-start,max_gpu_mb=torch.cuda.max_memory_allocated()/2**20);(out/'status.json').write_text(json.dumps(status,indent=2));print(json.dumps(status),flush=True)
 if step%2200==0 or step==a.steps:save(step)
(out/'complete.json').write_text(json.dumps(dict(steps=a.steps,elapsed_s=time.monotonic()-start)))
