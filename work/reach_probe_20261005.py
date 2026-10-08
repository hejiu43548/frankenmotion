import sys,json,hashlib
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/reach_demo_20261005';sys.path.insert(0,str(R/'work'))
import table_goal_adapter_20261005 as ga
from core import torch,np,FK,sample,quantity,HH
from generate import inputs
out=D/'command_probe';out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.12);m=ga.load(R/'outputs_amass/gait_demo_20261005/frozen/goal_adapter.pt');m.denoiser.goal=None;fk=FK('cuda');manifest=json.loads((ga.pa.BASE/'evaluation_manifest.json').read_text());rows=[]
for task,cs in [('reach',[.3,.4,.5]),('back_walk',[.35,.55]),('sidestep',[.4,.8])]:
 for pi in range(2):
  src=next(r for r in manifest if r['source']==f'{task}_p{pi}_s0');local,tx,cmd,controls=inputs(src,cs);seed=81005000+pi;raw=sample(m,local,tx,src['task_id'],cmd,[seed]*len(cs),True,root_controls=controls)
  with torch.no_grad():pos,poses,root=fk(raw,canonical=False,return_pose=True);canonical=fk(raw);q=quantity(canonical,task,HH/fk.height)
  for j,c in enumerate(cs):
   name=f'{task}_p{pi}_c{j}';p=out/(name+'.npz');np.savez_compressed(p,motion=raw[j].cpu().numpy(),joints_zup_m=pos[j].cpu().numpy(),poses_axisangle=poses[j].cpu().numpy(),root_translation=root[j].cpu().numpy(),human_height=fk.height,fps=20.,task=task,command=c)
   hand=canonical[j,:,21].cpu().numpy();pelvis=canonical[j,:,0].cpu().numpy();row=dict(name=name,task=task,command=c,seed=seed,path=str(p),human_quantity=float(q[j]),human_hand_height_p95=float(np.quantile(hand[:,2],.95)),human_hand_height_last=float(hand[-1,2]),human_hand_rel_last=(hand[-1]-pelvis[-1]).tolist(),root_delta=(pelvis[-1]-pelvis[0]).tolist());rows.append(row);print(row,flush=True)
(out/'manifest.json').write_text(json.dumps(rows,indent=2))
