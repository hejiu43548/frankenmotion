import sys,json
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/gait_demo_20261005';sys.path.insert(0,str(R/'work'));import table_goal_adapter_20261005 as ga
from core import torch,np,FK,sample
out=D/'generator_probe';out.mkdir(exist_ok=False);torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.08);model=ga.load(R/'outputs_amass/table_demo_20261005/frozen/goal_adapter.pt');fk=FK('cuda');sources=[r for r in json.loads((ga.pa.BASE/'evaluation_manifest.json').read_text()) if r['task']=='walk' and r['source'].endswith('_s0')];reports=[]
for pi,src in enumerate(sources[:3]):
 for use_task in [True,False]:
  goals=[[1.65,0.]];local,tx,cmd,control,g=ga.setup(src,goals);model.denoiser.goal=g;raw=sample(model,local,tx,10,cmd,[72005000],use_task,root_controls=control)
  with torch.no_grad():pts,poses,root=fk(raw,canonical=False,return_pose=True);_,detail=ga.metrics(raw,fk,g)
  xyz=pts[0].cpu().numpy();name=f'p{pi}_task{int(use_task)}';np.savez_compressed(out/(name+'.npz'),motion=raw[0].cpu().numpy(),joints_zup_m=xyz,poses_axisangle=poses[0].cpu().numpy(),root_translation=root[0].cpu().numpy(),human_height=fk.height,fps=20.)
  rec=dict(name=name,prompt=src['prompt'],use_task=use_task,**detail)
  for tag,idx in [('elbow',[16,17,18,19,20,21]),('knee',[1,2,4,5,7,8])]:
   a=xyz[:,idx[:2]]-xyz[:,idx[2:4]];b=xyz[:,idx[4:]]-xyz[:,idx[2:4]];v=180-np.degrees(np.arccos(np.clip((a*b).sum(-1)/np.linalg.norm(a,axis=-1)/np.linalg.norm(b,axis=-1),-1,1)));rec[tag+'_mean']=v[20:100].mean(0).tolist();rec[tag+'_std']=v[20:100].std(0).tolist()
  reports.append(rec);print(rec,flush=True)
(out/'results.json').write_text(json.dumps(reports,indent=2))
