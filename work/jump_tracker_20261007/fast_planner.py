"""Exploratory native CPU tracking of generated or source-disjoint mocap references.
Uses one exported shared actor, fixed flat scene, no reference adaptation or task routing.
Historical 11-task scores remain on their original GPU evaluation protocol.
"""
import os,json,argparse,hashlib,subprocess,sys,shutil
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
os.environ.setdefault('OMP_NUM_THREADS','1')
import numpy as np,mujoco,torch
from scipy.spatial.transform import Rotation
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/universal_tracker_20261006';W=R/'work/universal_tracker_20261006'
sys.path[:0]=[str(R/'work'),str(R/'outputs_amass/franken_eleven_20261003/code')]
from turn_anchor_20261005 import anchor_reference
import transfer as tr
from pulse_residual import PulseResidual
G={}
def initialize(actor_path,checkpoint,out,entry,anchors):
 torch.set_num_threads(1)
 template=D/'table_evaluation/baseline_stable/scene_000/reference_input';c=json.loads((template/'inference_contract.json').read_text());c['preview_offsets']=json.loads(Path(actor_path).with_suffix('.json').read_text())['preview_offsets']
 model=mujoco.MjModel.from_binary_path(str(template/'scene.mjb'))
 for bid in range(model.nbody):
  if model.body(bid).name=='demo_table':model.body_pos[bid,:2]=[100.,100.]
 assert model.body_pos[model.body('demo_table').id,0]==100.,'Table must be outside evaluation region'
 joints=[model.joint('robot/'+n).id for n in c['joint_names']];qa=model.jnt_qposadr[joints];va=model.jnt_dofadr[joints]
 contract=json.loads((D/'evaluation/baseline_broad_dev/contract.json').read_text());bodies=[model.body('robot/'+n).id for n in contract['body_names']]
 aids=[int(np.where(model.actuator_trnid[:,0]==model.joint('robot/'+n).id)[0][0]) for n in c['action_target_names']]
 source=tr.rt.load_model();source_names=[source.joint(i).name for i in range(1,source.njnt)];order=[source_names.index(n) for n in c['joint_names']]
 G.update(use_anchors=anchors,entry=entry,actor=torch.jit.load(actor_path,map_location='cpu').eval(),model=model,c=c,qa=qa,va=va,bodies=bodies,aids=aids,source=source,order=order,out=Path(out),checkpoint=checkpoint,anchor=model.body(c['anchor_body_name']).id,pelvis=model.body('robot/pelvis').id)
def quatrot(x):return Rotation.from_quat(np.asarray(x)[[1,2,3,0]])
def execute(row):
 try:return rollout(row)
 except Exception as exc:return dict(row,error=repr(exc),physical_complete=False)
def rollout(row):
 m=G['model'];c=G['c'];ref=dict(np.load(row['motion_path']));
 if G['entry']=='rsi':ref={k:v[50:] if k!='fps' else v for k,v in ref.items()}
 n=len(ref['joint_pos']);d=mujoco.MjData(m)
 sd=mujoco.MjData(G['source']);sd.qpos[:]=np.load(row['reference_path'])['reference_qpos'][0];sd.qpos[7:]=tr.rt.Q0;tr.rt.floor_align(G['source'],sd)
 d.qpos[:7]=sd.qpos[:7];d.qpos[G['qa']]=sd.qpos[7:][G['order']];d.qvel[:]=0
 if G['entry']=='rsi':
  d.qpos[:7]=np.r_[ref['body_pos_w'][0,0],ref['body_quat_w'][0,0]];d.qpos[G['qa']]=ref['joint_pos'][0]
  d.qvel[:6]=np.r_[ref['body_lin_vel_w'][0,0],quatrot(ref['body_quat_w'][0,0]).inv().apply(ref['body_ang_vel_w'][0,0])];d.qvel[G['va']]=ref['joint_vel'][0]
 mujoco.mj_forward(m,d)
 pulse=PulseResidual(ref,c,row['pulse_parameters'])
 initial_ref={k:v.copy() for k,v in ref.items()};anchor_events=[];anchor_frames=set(row.get('anchor_frames50',[])) if G['use_anchors'] else set();assert not anchor_frames or G['entry']=='standing'
 initial=d.qpos.copy();initial_v=d.qvel.copy();last=np.zeros(29);states=[initial];velocities=[initial_v];acts=[];controls=[];positions=[d.xpos[G['bodies']].copy()];body_quats=[d.xquat[G['bodies']].copy()];termination=None
 def sensor(name):
  s=m.sensor(name);return d.sensordata[s.adr[0]:s.adr[0]+s.dim[0]].copy()
 def observation(i):
  inv=quatrot(d.xquat[G['anchor']]).inv()
  def relative(k):return inv.apply(ref['body_pos_w'][k,c['reference_anchor_index']]-d.xpos[G['anchor']]),(inv*quatrot(ref['body_quat_w'][k,c['reference_anchor_index']])).as_matrix()[:,:2].reshape(-1)
  pos,rot=relative(i);parts=[ref['joint_pos'][i],ref['joint_vel'][i],pos,rot,sensor(c['linear_velocity_sensor']),sensor(c['angular_velocity_sensor']),d.qpos[G['qa']]-np.asarray(c['default_joint_pos']),d.qvel[G['va']],last]
  for offset in c['preview_offsets']:
   k=min(i+offset,n-1);pos,rot=relative(k);parts.extend([ref['joint_pos'][k],ref['joint_vel'][k],pos,rot])
  return np.concatenate(parts).astype(np.float32)
 with torch.inference_mode():
  for i in range(n-1):
   if i in anchor_frames:anchor_events.append(anchor_reference(ref,i,d.qpos.copy(),'relative_command_boundary'))
   act=G['actor'](torch.from_numpy(observation(i))[None])[0].numpy();last=act.copy()
   if c['clip_actions'] is not None:last=np.clip(last,-c['clip_actions'],c['clip_actions'])
   d.ctrl[G['aids']]=last*np.asarray(c['action_scale'])+np.asarray(c['action_offset'])
   pulse.apply(d,i,G['aids']);last=(d.ctrl[G['aids']]-np.asarray(c['action_offset']))/np.asarray(c['action_scale'])
   for _ in range(round(c['control_timestep']/m.opt.timestep)):mujoco.mj_step(m,d)
   mujoco.mj_forward(m,d);states.append(d.qpos.copy());velocities.append(d.qvel.copy());acts.append(act.copy());controls.append(d.ctrl.copy());positions.append(d.xpos[G['bodies']].copy());body_quats.append(d.xquat[G['bodies']].copy())
   w,x,y,z=d.qpos[3:7];roll=np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y));pitch=np.arcsin(np.clip(2*(w*y-z*x),-1,1))
   if not np.isfinite(d.qpos).all() or d.xpos[G['pelvis'],2]<.35 or np.hypot(roll,pitch)>np.pi/3:termination=(i+1)*.02;break
 states=np.asarray(states);positions=np.asarray(positions);observed=len(states);start=min(50 if G['entry']=='standing' else 0,observed-1);err=np.linalg.norm(positions[start:]-ref['body_pos_w'][start:observed],axis=-1)
 global_mpjpe=float(err.mean());root_idx=0;root_error=float(err[:,root_idx].mean());aligned=positions[start:]-positions[start:,root_idx:root_idx+1]-ref['body_pos_w'][start:observed]+ref['body_pos_w'][start:observed,root_idx:root_idx+1]
 scale=tr.HH/tr.robot_height(G['source']);height=float((states[start:,2].max()-states[start,2])*scale)
 feet=[G['bodies'].index(m.body('robot/'+side+'_ankle_roll_link').id) for side in ['left','right']];clear=(positions[start:,feet,2]-positions[start,feet,2]).min(-1);peak=int(clear.argmax());event=height>=.18 and clear[peak]>=.12/scale and bool(np.any(clear[peak:]<=.08/scale))
 loss=min(((height-row['command'])/.04)**2,100)+4*(not event)+100*(termination is not None)+.1*global_mpjpe
 final_tilt=float(np.hypot(roll,pitch));final_speed=float(np.linalg.norm(velocities[-1][:3]));loss+=5*(max(final_tilt-.2,0)/.2)**2+5*max(final_speed-.3,0)**2
 return dict(loss=float(loss),height=height,event=bool(event),complete=termination is None,termination=termination,final_tilt_rad=final_tilt,final_root_speed_mps=final_speed)
