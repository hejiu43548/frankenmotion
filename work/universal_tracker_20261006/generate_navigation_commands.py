"""New distance/direction, turn and wave commands from existing learned adapters.
CPU inference; new noise, explicit command-space refinement, no pose editing.
"""
import sys,json,hashlib,math
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/universal_tracker_20261006';sys.path.insert(0,str(R/'work'))
import table_goal_adapter_20261005 as ga
from core import torch,np,FK,sample
from generate import inputs
out=D/'navigation_commands';out.mkdir(exist_ok=False);torch.set_num_threads(4);weight=D/'backup/stable_frozen/goal_adapter.pt'
model,_=ga.pa.core.load_model('cpu',ga.BASE);model.denoiser=ga.GoalControl(model.denoiser);state=torch.load(weight,map_location='cpu',weights_only=False);missing=model.denoiser.load_state_dict(state['adapter'],strict=False);assert not missing.unexpected_keys and all(k.startswith('base.') for k in missing.missing_keys);model.eval();fk=FK('cpu')
sources=json.loads((ga.pa.BASE/'evaluation_manifest.json').read_text());templates={task:next(r for r in sources if r['source']==task+'_p0_s0') for task in ['walk','turn','wave']}
def setup_walk(distance,angle):
 g=torch.tensor([[distance,angle]],dtype=torch.float32);speed=g[:,0]/4.3*ga.pa.HH/fk.height;local,tx,cmd,_=inputs(templates['walk'],speed.tolist(),device='cpu');n=local.shape[1];tt=torch.arange(n)*.05;profile=((tt-.6)/.6).clamp(0,1)*((5.3-tt)/.8).clamp(0,1);profile/=profile[:-1].sum()*.05;values=torch.zeros(1,n,2);values[:,:,0]=g[:,0,None]*profile;values[:,:,1]=g[:,1,None]*(tt<1.2).float()/1.2;controls=ga.pa.core.encode_control(values,torch.ones(1,n,dtype=torch.bool));return local,tx,cmd,controls,g
routes=[ [('walk',1.0,0.),('turn',90.,0.),('walk',1.2,.2),('turn',90.,0.),('walk',.8,0.),('wave',.12,0.)], [('walk',1.4,-.25),('turn',135.,0.),('walk',1.0,.2),('turn',90.,0.),('walk',1.1,0.),('wave',.20,0.)] ];rows=[]
for ri,route in enumerate(routes):
 for si,(task,command,angle) in enumerate(route):
  seed=98061000+ri*100+si;history=[];model.denoiser.goal=None
  if task=='walk':
   human_distance=command*fk.height/1.0486437524221748;goal=np.array([human_distance,angle]);target=human_distance*np.array([np.cos(angle),np.sin(angle)]);best=None
   for iteration in range(5):
    local,tx,cmd,controls,g=setup_walk(*goal);model.denoiser.goal=g;raw=sample(model,local,tx,10,cmd,[seed],True,root_controls=controls)
    with torch.no_grad():p=fk(raw);xy=(p[0,-1,0,:2]-p[0,0,0,:2]).numpy();error=xy-target
    norm=float(np.linalg.norm(error));history.append(dict(injected_distance_human_m=float(goal[0]),injected_direction_rad=float(goal[1]),endpoint_error_human_m=norm))
    if best is None or norm<best[0]:best=(norm,raw.clone(),goal.copy())
    if norm<.015:break
    v=goal[0]*np.array([np.cos(goal[1]),np.sin(goal[1])])-error;goal=np.array([np.clip(np.linalg.norm(v),.8,2.8),np.clip(np.arctan2(v[1],v[0]),-.65,.65)])
   _,raw,injected=best;achieved=dict(endpoint_error_human_m=best[0],injected_goal=injected.tolist())
  elif task=='turn':
   target=np.radians(command);injected=target;best=None
   for iteration in range(5):
    local,tx,cmd,controls=inputs(templates['turn'],[injected],device='cpu');raw=sample(model,local,tx,4,cmd,[seed],True,root_controls=controls)
    with torch.no_grad():p=fk(raw).numpy()[0];side=p[:,1,:2]-p[:,2,:2];yaw=np.unwrap(np.arctan2(side[:,1],side[:,0]));actual=float(-(yaw[-1]-yaw[0]))
    error=abs(actual-target);history.append(dict(injected_rad=float(injected),generated_rad=actual))
    if best is None or error<best[0]:best=(error,raw.clone(),injected,actual)
    if error<np.radians(1):break
    injected=float(np.clip(injected+target-actual,.1,4.5))
   _,raw,injected,actual=best;achieved=dict(generated_turn_deg=float(np.degrees(actual)),injected_rad=injected)
  else:
   local,tx,cmd,controls=inputs(templates['wave'],[command],device='cpu');raw=sample(model,local,tx,3,cmd,[seed],True,root_controls=controls);achieved={}
  with torch.no_grad():pos,poses,root=fk(raw,canonical=False,return_pose=True)
  path=out/f'route_{ri}_stage_{si}_{task}.npz';np.savez_compressed(path,motion=raw[0].numpy(),joints_zup_m=pos[0].numpy(),poses_axisangle=poses[0].numpy(),root_translation=root[0].numpy(),human_height=fk.height,fps=20.)
  rows.append(dict(route=ri,stage=si,task=task,command=command,direction_rad=angle,seed=seed,path=str(path),source=f'route_{ri}_stage_{si}',prompt=templates[task]['prompt'],refinements=history,achieved=achieved,units='robot distance m and relative direction rad' if task=='walk' else 'right turn degrees' if task=='turn' else 'human-equivalent wave amplitude m',generator_goal_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),human_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
  (out/'manifest.json').write_text(json.dumps(rows,indent=2));print(ri,si,task,command,achieved,flush=True)
(out/'protocol.json').write_text(json.dumps(dict(routes=routes,scope='Development relative-command sequences. Existing FrankenMotion learned command adapters, new noise. Numerical input refinement only; GMR and sequence assembly handled separately. No hand-written joint animation.',goal_adapter=str(weight),physical_task_adapter=str(ga.BASE)),indent=2))
