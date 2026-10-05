"""Train one actor from offline mixed teachers; no router is retained in the student."""
import json,time,hashlib,argparse,copy
from pathlib import Path
import numpy as np,torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/gait_demo_20261005';p=argparse.ArgumentParser();p.add_argument('--name',default='distill_v1');p.add_argument('--steps',type=int,default=3000);p.add_argument('--dataset',default='teacher_data/dataset.npz');p.add_argument('--initial',default=str(R/'outputs_amass/table_demo_20261005/frozen/policy.pt'));a=p.parse_args();out=D/'training'/a.name;out.mkdir(exist_ok=False);torch.set_num_threads(2);torch.manual_seed(74005001);torch.cuda.set_per_process_memory_fraction(.06);data=np.load(D/a.dataset);obs=torch.tensor(data['observations'],device='cuda');actions=torch.tensor(data['actions'],device='cuda');groups=torch.tensor(data['group'],device='cuda');blend=torch.tensor(data['blend'],device='cuda');last=int(groups.max());train=groups<last-1;test=~train;walk=torch.where(train&(blend<.1))[0];reach=torch.where(train&(blend>.9))[0];transition=torch.where(train&(blend>=.1)&(blend<=.9))[0];assert len(walk)>0 and len(reach)>0;dummy=TensorDict({'actor':torch.zeros(1,361,device='cuda')},batch_size=[1]);actor=MLPModel(dummy,{'actor':['actor']},'actor',29,hidden_dims=[512,256,128],activation='elu',obs_normalization=True,stochastic=True).cuda();ck=torch.load(a.initial,map_location='cpu',weights_only=False);actor.load_state_dict(ck['actor_state_dict']);actor.eval();opt=torch.optim.AdamW(actor.parameters(),lr=3e-5,weight_decay=1e-5);weights=torch.ones(29,device='cuda');weights[:12]=3.;weights[12:15]=2.;records=[];start=time.time()
def save(step):
 outck=copy.deepcopy(ck);outck['actor_state_dict']={k:v.detach().cpu() for k,v in actor.state_dict().items()};outck['distillation']=dict(step=step,source=a.initial,dataset=str(D/a.dataset),single_actor=True,teachers_only_during_training=True);path=out/f'model_{step}.pt';torch.save(outck,path);path.with_suffix('.pt.ready.json').write_text(json.dumps(dict(sha256=hashlib.sha256(path.read_bytes()).hexdigest())))
for step in range(a.steps+1):
 if step%500==0 or step==a.steps:
  with torch.no_grad():pred=actor(TensorDict({'actor':obs[test]},batch_size=[int(test.sum())]));error=pred-actions[test];rec=dict(step=step,validation_mse=float(error.square().mean()),validation_leg_mse=float(error[:,:12].square().mean()),elapsed_s=time.time()-start)
  records.append(rec);(out/'progress.json').write_text(json.dumps(records,indent=2));save(step);print(rec,flush=True)
  if step==a.steps:break
 ids=torch.cat([g[torch.randint(len(g),(n,),device='cuda')] for g,n in [(walk,256),(reach,192),(transition,64)]]);pred=actor(TensorDict({'actor':obs[ids]},batch_size=[len(ids)]));loss=((pred-actions[ids]).square()*weights).mean();opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(actor.parameters(),1.);opt.step()
(out/'complete.json').write_text(json.dumps(dict(steps=a.steps,one_actor=True,no_routing_at_inference=True,heldout_teacher_groups=[last-1,last],source_dataset_sha256=hashlib.sha256((D/a.dataset).read_bytes()).hexdigest()),indent=2))
