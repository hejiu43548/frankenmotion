"""Fresh paired table sequences after checkpoint freeze. CPU diffusion only.
All scene layouts are sampled before any tracking; same-noise input refinement.
"""
import sys,json,hashlib
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/universal_tracker_20261006';sys.path.insert(0,str(R/'work'))
import reach_adapter_v6_20261005 as ra
from core import torch,np,FK,sample
from generate import inputs
ga=ra.ga
selection=json.loads((D/'frozen_unified/protocol.json').read_text());assert hashlib.sha256(Path(selection['checkpoint']).read_bytes()).hexdigest()==selection['checkpoint_sha256'];out=D/'fresh_table';out.mkdir(exist_ok=False);torch.set_num_threads(4);fk=FK('cpu');goalweight=D/'backup/stable_frozen/goal_adapter.pt';reachweight=D/'backup/stable_frozen/reach_adapter.pt'
m,_=ga.pa.core.load_model('cpu',ga.BASE);m.denoiser=ga.GoalControl(m.denoiser);state=torch.load(goalweight,map_location='cpu',weights_only=False);r=m.denoiser.load_state_dict(state['adapter'],strict=False);assert not r.unexpected_keys and all(k.startswith('base.') for k in r.missing_keys);m.eval();sources=json.loads((ga.pa.BASE/'evaluation_manifest.json').read_text());templates={task:next(r for r in sources if r['source']==task+'_p0_s0') for task in ['walk','turn']}
rng=np.random.default_rng(107061000);layouts=[dict(distance_robot_m=float(rng.uniform(.85,1.75)),direction_rad=float(rng.uniform(-.5,.5))) for i in range(6)];(out/'layouts.json').write_text(json.dumps(layouts,indent=2))
def save(raw,path,task,command):
 with torch.no_grad():pos,poses,root=fk(raw,canonical=False,return_pose=True)
 np.savez_compressed(path,motion=raw[0].numpy(),joints_zup_m=pos[0].numpy(),poses_axisangle=poses[0].numpy(),root_translation=root[0].numpy(),human_height=fk.height,fps=20.,task=task,command=command)
def setup_goal(goal):
 g=torch.tensor([goal],dtype=torch.float32);speed=g[:,0]/4.3*ga.pa.HH/fk.height;local,tx,cmd,_=inputs(templates['walk'],speed.tolist(),device='cpu');n=local.shape[1];tt=torch.arange(n)*.05;profile=((tt-.6)/.6).clamp(0,1)*((5.3-tt)/.8).clamp(0,1);profile/=profile[:-1].sum()*.05;values=torch.zeros(1,n,2);values[:,:,0]=g[:,0,None]*profile;values[:,:,1]=g[:,1,None]*(tt<1.2).float()/1.2;controls=ga.pa.core.encode_control(values,torch.ones(1,n,dtype=torch.bool));return local,tx,cmd,controls,g
for phase,base in [('walk',107061000),('away',107063000)]:
 folder=out/phase;folder.mkdir();rows=[]
 for i,layout in enumerate(layouts):
  distance=layout['distance_robot_m'] if phase=='walk' else 1.2;angle=layout['direction_rad'] if phase=='walk' else 0.;seed=base+i;hd=distance*fk.height/1.0486437524221748;target=hd*np.array([np.cos(angle),np.sin(angle)]);goal=[hd,angle];history=[];best=None
  for iteration in range(4):
   local,tx,cmd,controls,g=setup_goal(goal);m.denoiser.goal=g;raw=sample(m,local,tx,10,cmd,[seed],True,root_controls=controls)
   with torch.no_grad():p=fk(raw);xy=(p[0,-1,0,:2]-p[0,0,0,:2]).numpy();error=xy-target
   norm=float(np.linalg.norm(error));history.append(dict(iteration=iteration,injected_distance_human_m=float(goal[0]),injected_direction_rad=float(goal[1]),endpoint_error_human_m=norm,achieved_xy_human_m=xy.tolist()))
   if best is None or norm<best[0]:best=(norm,raw.clone(),iteration,float(cmd[0]),xy.copy())
   if norm<.015:break
   v=goal[0]*np.array([np.cos(goal[1]),np.sin(goal[1])])-error;goal=[float(np.clip(np.linalg.norm(v),.8,2.8)),float(np.clip(np.arctan2(v[1],v[0]),-.65,.65))]
  dest=folder/f'scene_{i:03d}';dest.mkdir();save(best[1],dest/'human_walk.npz','walk',best[3]);forward=np.array([np.cos(angle),np.sin(angle)]);right=np.array([np.sin(angle),-np.cos(angle)]);goalxy=distance*forward;center=goalxy+.65*forward;hand=goalxy+.4*forward+.14*right
  row=dict(index=i,seed=seed,distance_robot_m=distance,direction_rad=angle,distance_human_m=hd,goal_xy=goalxy.tolist(),table_center=[*center.tolist(),.4],table_half_size=[.4,.48,.035],table_top=.8,table_yaw=angle,hand_target=[*hand.tolist(),.84],goal_generation=dict(endpoint_error_m=[best[0]],xy=[best[4].tolist()]),generation_refinements=history,selected_generation_iteration=best[2],weight_sha256=hashlib.sha256(goalweight.read_bytes()).hexdigest(),source=str(dest),prompt=templates['walk']['prompt']);(dest/'scene.json').write_text(json.dumps(row,indent=2));rows.append(row);(folder/'manifest.json').write_text(json.dumps(rows,indent=2));print(phase,i,flush=True)
folder=out/'turn';folder.mkdir();rows=[];m.denoiser.goal=None
for i in range(6):
 for degree in [90,135]:
  target=np.radians(degree);injected=target;best=None;history=[];seed=107064000+i
  for iteration in range(4):
   local,tx,cmd,controls=inputs(templates['turn'],[injected],device='cpu');raw=sample(m,local,tx,4,cmd,[seed],True,root_controls=controls)
   with torch.no_grad():p=fk(raw).numpy()[0];side=p[:,1,:2]-p[:,2,:2];yaw=np.unwrap(np.arctan2(side[:,1],side[:,0]));actual=float(-(yaw[-1]-yaw[0]))
   err=abs(actual-target);history.append(dict(injected_rad=float(injected),generated_rad=actual))
   if best is None or err<best[0]:best=(err,raw.clone(),injected,actual)
   if err<np.radians(1):break
   injected=float(np.clip(injected+target-actual,0,4.5))
  path=folder/f'turn_{seed}_{degree}.npz';save(best[1],path,'turn',target);rows.append(dict(layout=i,seed=seed,command_deg=degree,path=str(path),generated_deg=float(np.degrees(best[3])),injected_rad=best[2],refinements=history));(folder/'results.json').write_text(json.dumps(rows,indent=2));print('turn',i,degree,flush=True)
m.denoiser=ra.ReachControl(m.denoiser);state=torch.load(reachweight,map_location='cpu',weights_only=False);r=m.denoiser.load_state_dict(state['adapter'],strict=False);assert not r.unexpected_keys and all(k.startswith('base.') for k in r.missing_keys);m.eval();folder=out/'reach';folder.mkdir();rows=[];walks=json.loads((out/'walk/manifest.json').read_text())
for i in range(12):
 command=[.3,.5][i%2];seed=107062000+i//2;src=dict(prompt=str(ra.D/'prompts/reach_place_p0.json'),task='reach',frames=120);local,tx,cmd,_=inputs(src,[command],device='cpu');m.denoiser.command=torch.tensor([[command,.84]],dtype=torch.float32);raw=sample(m,local,tx,1,cmd,[seed],True,steps=50);dest=folder/f'scene_{i:03d}';dest.mkdir();save(raw,dest/'human_reach.npz','reach',command);rows.append(dict(index=i,source=str(dest),command=command,exit_task='none',exit_command=0.,seed=seed,walk_source=walks[i//2]['source'],generator_weight=str(reachweight),generator_sha256=hashlib.sha256(reachweight.read_bytes()).hexdigest()));(folder/'manifest.json').write_text(json.dumps(rows,indent=2));print('reach',i,command,flush=True)
(out/'generation_protocol.json').write_text(json.dumps(dict(selection_sha256=hashlib.sha256((D/'frozen_unified/protocol.json').read_bytes()).hexdigest(),layouts=6,paired_sequences=12,seed_bases=dict(walk=107061000,reach=107062000,away=107063000,turn=107064000),scope=__doc__,fixed_table_height_m=.8,reach_units='human-equivalent wrist reach',turn_extrapolation='90/135deg exceed nominal0.45-1.5rad training range; explicit same-noise command input refinement, no turn pose edits',goal_sha256=hashlib.sha256(goalweight.read_bytes()).hexdigest(),reach_sha256=hashlib.sha256(reachweight.read_bytes()).hexdigest()),indent=2))
