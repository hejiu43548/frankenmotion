"""Single shared student, frozen observation normalization, training-only teachers."""
import json,time,argparse,copy,hashlib
from pathlib import Path
import numpy as np,torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/turn_demo_20261005';p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--datasets',nargs='+',required=True);p.add_argument('--initial',default=str(R/'outputs_amass/gait_demo_20261005/frozen/policy.pt'));p.add_argument('--steps',type=int,default=4000);a=p.parse_args();out=D/a.name;out.mkdir(exist_ok=False);torch.set_num_threads(2);torch.manual_seed(82005000);torch.cuda.set_per_process_memory_fraction(.08)
zs=[np.load(D/x/'dataset.npz') for x in a.datasets];v={k:np.concatenate([z[k] for z in zs]) for k in ['observations','actions','group','blend']};obs=torch.tensor(v['observations'],dtype=torch.float32,device='cuda');target=torch.tensor(v['actions'],dtype=torch.float32,device='cuda');train=torch.tensor(v['group']<42,device='cuda');blend=torch.tensor(v['blend'],device='cuda');test=~train;sets=[torch.where(train&(blend<.05))[0],torch.where(train&(blend>.95))[0],torch.where(train&(blend>=.05)&(blend<=.95))[0]]
dummy=TensorDict({'actor':torch.zeros(1,361,device='cuda')},batch_size=[1]);actor=MLPModel(dummy,{'actor':['actor']},'actor',29,hidden_dims=[512,256,128],activation='elu',obs_normalization=True,stochastic=True).cuda();ck=torch.load(a.initial,map_location='cpu',weights_only=False);actor.load_state_dict(ck['actor_state_dict']);actor.eval();opt=torch.optim.AdamW(actor.parameters(),lr=3e-5,weight_decay=1e-5);w=torch.ones(29,device='cuda');w[:12]=2.;start=time.time();records=[]
def predict(ids):return actor(TensorDict({'actor':obs[ids]},batch_size=[len(ids)]))
for step in range(a.steps+1):
 if step%500==0:
  with torch.no_grad():
   ids=torch.where(test)[0];pred=predict(ids);err=(pred-target[ids]).square();row=dict(step=step,validation_mse=float(err.mean()),walk_mse=float(err[blend[ids]<.05].mean()),reach_mse=float(err[blend[ids]>.95].mean()),elapsed_s=time.time()-start)
  records.append(row);(out/'progress.json').write_text(json.dumps(records,indent=2));outck=copy.deepcopy(ck);outck['actor_state_dict']={k:v.detach().cpu() for k,v in actor.state_dict().items()};outck['distillation']=dict(step=step,datasets=a.datasets,initial=a.initial,training_groups=list(range(42)),validation_groups=list(range(42,48)),deployment='One fixed actor, no action routing or mixture');path=out/f'model_{step}.pt';torch.save(outck,path);path.with_suffix('.pt.ready.json').write_text(json.dumps(dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest())));print(row,flush=True)
  if step==a.steps:break
 ids=torch.cat([g[torch.randint(len(g),(n,),device='cuda')] for g,n in zip(sets,[256,224,32])]);loss=((predict(ids)-target[ids]).square()*w).mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(actor.parameters(),1.);opt.step()
(out/'complete.json').write_text(json.dumps(dict(steps=a.steps,one_fixed_actor=True,teachers_only_in_training=True)))
