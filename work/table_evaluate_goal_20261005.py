"""Matched-noise free generation: old root controls vs learned goal branch."""
import sys,json,hashlib
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/table_demo_20261005';sys.path.insert(0,str(R/'work'))
import table_goal_adapter_20261005 as ga
from core import torch,np,FK,sample
out=D/'goal_command_audit';out.mkdir(exist_ok=False);torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.1);weight=D/'goal_v2/best.pt';model=ga.load(weight);fk=FK('cuda');scale=1.0486437524221748/fk.height;sources=[r for r in json.loads((ga.pa.BASE/'evaluation_manifest.json').read_text()) if r['task']=='walk' and r['source'].endswith('_s0')];records=[];traces=[]
for pi,src in enumerate(sources):
 for si in range(2):
  goals=[[d,angle] for d in [1.,1.8,2.6] for angle in [-.5,-.25,0.,.25,.5]];local,tx,cmd,control,g=ga.setup(src,goals);seed=56005000+pi*100+si
  for kind in ['existing_speed_yaw','learned_goal']:
   model.denoiser.goal=g if kind=='learned_goal' else None;raw=sample(model,local,tx,10,cmd,[seed]*len(goals),True,root_controls=control)
   with torch.no_grad():pts=fk(raw)[:,:,0,:2];xy=(pts-pts[:,:1])*scale
   for i,(dist,angle) in enumerate(goals):
    path=xy[i].cpu().numpy();target=dist*scale*np.array([np.cos(angle),np.sin(angle)]);end=path[-1];records.append(dict(kind=kind,prompt=pi,seed=seed,distance_robot_m=dist*scale,direction_rad=angle,actual_xy=end.tolist(),actual_distance_m=float(np.linalg.norm(end)),actual_direction_rad=float(np.arctan2(end[1],end[0])),endpoint_error_m=float(np.linalg.norm(end-target))));traces.append(path)
   print(pi,si,kind,flush=True)
summary={kind:dict(n=sum(r['kind']==kind for r in records),mean_endpoint_error_m=float(np.mean([r['endpoint_error_m'] for r in records if r['kind']==kind])),p90_endpoint_error_m=float(np.quantile([r['endpoint_error_m'] for r in records if r['kind']==kind],.9))) for kind in ['existing_speed_yaw','learned_goal']}
(out/'results.json').write_text(json.dumps(records,indent=2));np.savez_compressed(out/'paths.npz',xy=np.asarray(traces));(out/'audit.json').write_text(json.dumps(dict(summary=summary,weight_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),paired_seed_and_prompt=True,identical_legacy_speed_yaw_controls=True,source_motion_input=False,output_editing=False,scope='Fresh noise within cached prompt templates and new goal ranges. This ablates the goal branch with identical demo-derived speed/yaw profiles; some low-speed scalar commands lie below the old task adapter validated range. Not a general ranking of old model capability; endpoint metric only, no physical execution claim.'),indent=2));print(summary,flush=True)
