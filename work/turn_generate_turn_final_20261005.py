import sys,json
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/turn_demo_20261005';sys.path.insert(0,str(R/'work'))
import table_goal_adapter_20261005 as ga
from core import torch,np,FK,sample
from generate import inputs
torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.12)
m=ga.load(R/'outputs_amass/gait_demo_20261005/frozen/goal_adapter.pt');fk=FK('cuda');src=next(r for r in json.loads((ga.pa.BASE/'evaluation_manifest.json').read_text()) if r['source']=='turn_p0_s0');out=D/'turn_final';out.mkdir(exist_ok=False);rows=[]
for seed in range(93005000,93005006):
 cs=np.radians([90,135]).tolist();injected=np.asarray(cs);history=[]
 for iteration in range(4):
  local,tx,cmd,controls=inputs(src,injected.tolist());m.denoiser.goal=None;raw=sample(m,local,tx,4,cmd,[seed]*len(cs),True,root_controls=controls)
  with torch.no_grad():pp=fk(raw).cpu().numpy()
  side=pp[:,:,1,:2]-pp[:,:,2,:2];yy=np.unwrap(np.arctan2(side[:,:,1],side[:,:,0]),axis=1);actual=-(yy[:,-1]-yy[:,0]);history.append(dict(injected_rad=injected.tolist(),generated_rad=actual.tolist()))
  if iteration<3:injected=np.clip(injected+np.asarray(cs)-actual,0,4.5)
 with torch.no_grad():pos,poses,root=fk(raw,canonical=False,return_pose=True)
 for j,c in enumerate(cs):
  p=pos[j].cpu().numpy();side=p[:,1,:2]-p[:,2,:2];yaw=np.unwrap(np.arctan2(side[:,1],side[:,0]));actual=-(yaw[-1]-yaw[0]);path=out/f'turn_{seed}_{round(np.degrees(c))}.npz';np.savez_compressed(path,motion=raw[j].cpu().numpy(),joints_zup_m=p,poses_axisangle=poses[j].cpu().numpy(),root_translation=root[j].cpu().numpy(),fps=20.,human_height=fk.height,task='turn',command=c)
  rows.append(dict(seed=seed,command_deg=float(np.degrees(c)),generated_deg=float(np.degrees(actual)),path=str(path),injected_rad=float(injected[j]),refinements=[dict(injected_rad=h['injected_rad'][j],generated_rad=h['generated_rad'][j]) for h in history],root_displacement_m=float(np.linalg.norm(p[-1,0,:2]-p[0,0,:2]))))
 (out/'results.json').write_text(json.dumps(rows,indent=2));print(rows[-3:],flush=True)
