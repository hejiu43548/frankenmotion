import sys,json
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/reach_demo_20261005';sys.path.insert(0,str(R/'work'));import table_goal_adapter_20261005 as ga
from core import torch,np,FK,sample,quantity,HH,encode_control
from generate import inputs
out=D/'exit_style_probe';out.mkdir(exist_ok=False);torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.13);m=ga.load(R/'outputs_amass/gait_demo_20261005/frozen/goal_adapter.pt');fk=FK('cuda');sources=json.loads((ga.pa.BASE/'evaluation_manifest.json').read_text());rows=[]
for task,c in [('back_walk',.35),('sidestep',.5)]:
 for pi in range(2):
  src=next(r for r in sources if r['source']==f'{task}_p{pi}_s0');local,tx,cmd,controls=inputs(src,[c,c]);seeds=[89005000,89005001]
  for variant in ['vanilla','task_only','current','signed_root'] if task=='back_walk' else ['vanilla','current']:
   control=controls if variant=='current' else None
   if variant=='signed_root':
    v=torch.zeros(2,120,2,device='cuda');v[...,0]=-c*fk.height/HH;control=encode_control(v,torch.ones(2,120,device='cuda',dtype=torch.bool))
   raw=sample(m,local,tx,src['task_id'],cmd,seeds,variant!='vanilla',steps=50,root_controls=control)
   with torch.no_grad():
    pp,poses,root=fk(raw,canonical=False,return_pose=True);p=fk(raw);q=quantity(p,task,HH/fk.height);hips=p[:,:,[1,2]];knees=p[:,:,[4,5]];ankles=p[:,:,[7,8]];u=hips-knees;v=ankles-knees;angle=torch.acos(((u*v).sum(-1)/(u.norm(dim=-1)*v.norm(dim=-1))).clamp(-1,1));ks=angle.std(1)*180/np.pi;frange=ankles[:,:,:,2].quantile(.95,dim=1)-ankles[:,:,:,2].quantile(.05,dim=1)
   for j in range(2):
    name=f'{task}_p{pi}_{variant}_{j}';np.savez_compressed(out/(name+'.npz'),motion=raw[j].cpu().numpy(),joints_zup_m=pp[j].cpu().numpy(),poses_axisangle=poses[j].cpu().numpy(),root_translation=root[j].cpu().numpy(),human_height=fk.height,fps=20.,task=task,command=c);row=dict(name=name,task=task,prompt=pi,variant=variant,seed=seeds[j],quantity=float(q[j]),knee_std_deg=ks[j].tolist(),ankle_height_range_m=frange[j].tolist());rows.append(row);print(row,flush=True)
(out/'manifest.json').write_text(json.dumps(rows,indent=2))
