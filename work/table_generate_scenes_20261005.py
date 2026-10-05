import sys,json,argparse,hashlib
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/table_demo_20261005';sys.path.insert(0,str(R/'work'))
import table_goal_adapter_20261005 as ga
from core import torch,np,FK,sample
p=argparse.ArgumentParser();p.add_argument('--weight',required=True);p.add_argument('--name',required=True);p.add_argument('--count',type=int,default=6);p.add_argument('--seed',type=int,default=53005000);p.add_argument('--all-random',action='store_true');p.add_argument('--refine-iterations',type=int,default=0);p.add_argument('--layout-json');a=p.parse_args();out=D/a.name;out.mkdir(exist_ok=False);torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.12)
weight=Path(a.weight);(out/'goal_adapter.pt').write_bytes(weight.read_bytes());model=ga.load(out/'goal_adapter.pt');fk=FK('cuda');rng=np.random.default_rng(a.seed);rows=[]
layouts=json.loads(Path(a.layout_json).read_text()) if a.layout_json else None
if layouts is not None:assert len(layouts)==a.count
sources=json.loads((ga.pa.BASE/'evaluation_manifest.json').read_text());src=next(r for r in sources if r['source']=='walk_p0_s0');reachsrc=next(r for r in sources if r['source']=='reach_p0_s0')
# Random scene sampled before any generated motion or tracker execution.
for i in range(a.count):
 distance=float(rng.uniform(.85,1.75));angle=float(rng.uniform(-.5,.5))
 if layouts is not None:distance=float(layouts[i]['distance_robot_m']);angle=float(layouts[i]['direction_rad'])
 elif i==0 and not a.all_random:distance=1.2;angle=0.
 assert .6<distance<2.15 and abs(angle)<=.65
 human_distance=distance*fk.height/1.0486437524221748 # exact robot marker height checked by retarget stage
 goal=[[human_distance,angle]];original_g=torch.tensor(goal,device='cuda');targetxy=human_distance*np.array([np.cos(angle),np.sin(angle)]);refinements=[];best_error=float('inf');best_raw=None;best_iteration=None;best_command=None
 for iteration in range(a.refine_iterations+1):
  local,tx,cmd,controls,g=ga.setup(src,goal);model.denoiser.goal=g;candidate=sample(model,local,tx,10,cmd,[a.seed+i],True,root_controls=controls)
  with torch.no_grad():_,detail=ga.metrics(candidate,fk,original_g)
  error=np.array(detail['xy'][0])-targetxy;norm=float(np.linalg.norm(error));refinements.append(dict(iteration=iteration,injected_distance_human_m=goal[0][0],injected_direction_rad=goal[0][1],endpoint_error_human_m=norm,achieved_xy_human_m=detail['xy'][0]))
  if norm<best_error:best_error=norm;best_raw=candidate.clone();best_iteration=iteration;best_command=float(cmd[0])
  if norm<.015:break
  injected=goal[0][0]*np.array([np.cos(goal[0][1]),np.sin(goal[0][1])]);updated=injected-error;goal=[[float(np.clip(np.linalg.norm(updated),.8,2.8)),float(np.clip(np.arctan2(updated[1],updated[0]),-.65,.65))]]
 raw=best_raw
 with torch.no_grad():pos,poses,root=fk(raw,canonical=False,return_pose=True);_,metrics=ga.metrics(raw,fk,original_g)
 dest=out/f'scene_{i:03d}';dest.mkdir();np.savez_compressed(dest/'human_walk.npz',motion=raw[0].cpu().numpy(),joints_zup_m=pos[0].cpu().numpy(),poses_axisangle=poses[0].cpu().numpy(),root_translation=root[0].cpu().numpy(),human_height=fk.height,fps=20.,task='walk',command=best_command)
 model.denoiser.goal=None;local,tx,cmd,controls=ga.inputs(reachsrc,[.45]);raw=sample(model,local,tx,1,cmd,[a.seed+i],True,root_controls=controls)
 with torch.no_grad():pos,poses,root=fk(raw,canonical=False,return_pose=True)
 np.savez_compressed(dest/'human_reach.npz',motion=raw[0].cpu().numpy(),joints_zup_m=pos[0].cpu().numpy(),poses_axisangle=poses[0].cpu().numpy(),root_translation=root[0].cpu().numpy(),human_height=fk.height,fps=20.,task='reach',command=.45)
 forward=np.array([np.cos(angle),np.sin(angle)]);right=np.array([np.sin(angle),-np.cos(angle)]);goalxy=distance*forward;center=goalxy+.65*forward;target=goalxy+.4*forward+.14*right
 row=dict(index=i,seed=a.seed+i,distance_robot_m=distance,direction_rad=angle,distance_human_m=human_distance,goal_xy=goalxy.tolist(),table_center=[*center.tolist(),.4],table_half_size=[.4,.48,.035],table_top=.8,table_yaw=angle,hand_target=[*target.tolist(),.84],goal_generation=metrics,generation_refinements=refinements,selected_generation_iteration=best_iteration,weight_sha256=hashlib.sha256((out/'goal_adapter.pt').read_bytes()).hexdigest(),source=str(dest),prompt=src['prompt'])
 (dest/'scene.json').write_text(json.dumps(row,indent=2));rows.append(row);print('generated',i,metrics,flush=True)
(out/'manifest.json').write_text(json.dumps(rows,indent=2))
