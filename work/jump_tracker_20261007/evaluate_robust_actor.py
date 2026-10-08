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
from robustness_profiles import PROFILES,apply_model_profile,description
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/universal_tracker_20261006';W=R/'work/universal_tracker_20261006'
sys.path[:0]=[str(R/'work'),str(R/'outputs_amass/franken_eleven_20261003/code')]
from turn_anchor_20261005 import anchor_reference
import transfer as tr
G={}
def initialize(actor_path,checkpoint,out,entry,anchors,profile):
 torch.set_num_threads(1)
 template=D/'table_evaluation/baseline_stable/scene_000/reference_input';c=json.loads((template/'inference_contract.json').read_text());c['preview_offsets']=json.loads(Path(actor_path).with_suffix('.json').read_text())['preview_offsets']
 model=mujoco.MjModel.from_binary_path(str(template/'scene.mjb'))
 for bid in range(model.nbody):
  if model.body(bid).name=='demo_table':model.body_pos[bid,:2]=[100.,100.]
 assert model.body_pos[model.body('demo_table').id,0]==100.,'Table must be outside evaluation region'
 apply_model_profile(model,profile)
 joints=[model.joint('robot/'+n).id for n in c['joint_names']];qa=model.jnt_qposadr[joints];va=model.jnt_dofadr[joints]
 contract=json.loads((D/'evaluation/baseline_broad_dev/contract.json').read_text());bodies=[model.body('robot/'+n).id for n in contract['body_names']]
 aids=[int(np.where(model.actuator_trnid[:,0]==model.joint('robot/'+n).id)[0][0]) for n in c['action_target_names']]
 source=tr.rt.load_model();source_names=[source.joint(i).name for i in range(1,source.njnt)];order=[source_names.index(n) for n in c['joint_names']]
 G.update(profile=profile,use_anchors=anchors,entry=entry,actor=torch.jit.load(actor_path,map_location='cpu').eval(),model=model,c=c,qa=qa,va=va,bodies=bodies,aids=aids,source=source,order=order,out=Path(out),checkpoint=checkpoint,anchor=model.body(c['anchor_body_name']).id,pelvis=model.body('robot/pelvis').id)
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
 initial_ref={k:v.copy() for k,v in ref.items()};anchor_events=[];anchor_frames=set(row.get('anchor_frames50',[])) if G['use_anchors'] else set();assert not anchor_frames or G['entry']=='standing'
 delayed_target=np.asarray(c['action_offset']).copy();initial=d.qpos.copy();initial_v=d.qvel.copy();last=np.zeros(29);states=[initial];velocities=[initial_v];acts=[];controls=[];positions=[d.xpos[G['bodies']].copy()];body_quats=[d.xquat[G['bodies']].copy()];termination=None
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
   target=last*np.asarray(c['action_scale'])+np.asarray(c['action_offset']);d.ctrl[G['aids']]=delayed_target if G['profile']=='delay_20ms' else target;delayed_target=target.copy()
   d.xfrc_applied[:]=0
   if G['profile']=='lateral_push_40N' and 100<=i<105:d.xfrc_applied[G['pelvis'],1]=40.
   for _ in range(round(c['control_timestep']/m.opt.timestep)):mujoco.mj_step(m,d)
   mujoco.mj_forward(m,d);states.append(d.qpos.copy());velocities.append(d.qvel.copy());acts.append(act.copy());controls.append(d.ctrl.copy());positions.append(d.xpos[G['bodies']].copy());body_quats.append(d.xquat[G['bodies']].copy())
   w,x,y,z=d.qpos[3:7];roll=np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y));pitch=np.arcsin(np.clip(2*(w*y-z*x),-1,1))
   if not np.isfinite(d.qpos).all() or d.xpos[G['pelvis'],2]<.35 or np.hypot(roll,pitch)>np.pi/3:termination=(i+1)*.02;break
 states=np.asarray(states);positions=np.asarray(positions);observed=len(states);start=min(50 if G['entry']=='standing' else 0,observed-1);err=np.linalg.norm(positions[start:]-ref['body_pos_w'][start:observed],axis=-1)
 global_mpjpe=float(err.mean());root_idx=0;root_error=float(err[:,root_idx].mean());aligned=positions[start:]-positions[start:,root_idx:root_idx+1]-ref['body_pos_w'][start:observed]+ref['body_pos_w'][start:observed,root_idx:root_idx+1]
 dest=G['out']/Path(row['path']).stem;dest.mkdir(exist_ok=False);(dest/'anchor_events.json').write_text(json.dumps(anchor_events,indent=2));np.savez_compressed(dest/'initial_motion.npz',**initial_ref);np.savez_compressed(dest/'actual.npz',qpos=states,qvel=velocities,actions=acts,ctrl=controls,body_pos_w=positions,body_quat_w=body_quats,fps=50.)
 cc=dict(c,initial_qpos=initial.tolist(),initial_qvel=initial_v.tolist());(dest/'inference_contract.json').write_text(json.dumps(cc,indent=2));os.link(G['out']/'scene.mjb',dest/'scene.mjb');np.savez_compressed(dest/'motion.npz',**ref)
 result=dict(row,robustness_profile=G['profile'],causal_reference_anchoring=G['use_anchors'],entry=G['entry'],physical_complete=termination is None,termination_time=termination,observed_frames=observed,planned_frames=n,global_mpjpe_m=global_mpjpe,root_translation_error_m=root_error,root_aligned_mpjpe_m=float(np.linalg.norm(aligned,axis=-1).mean()),joint_rmse_rad=float(np.sqrt(np.mean((states[start:,G['qa']]-ref['joint_pos'][start:observed])**2))),reference_min_pelvis_height_m=float(ref['body_pos_w'][start:,0,2].min()),checkpoint=G['checkpoint'],run=str(dest),scope='Exploratory fixed native CPU scene, no state reset after initialization, one shared actor, optional explicitly logged future-reference anchoring at command boundaries, no semantic success inferred from survival. Errors over observed horizon only; fallen cases never counted successful.')
 (dest/'result.json').write_text(json.dumps(result,indent=2));return result
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--profile',choices=PROFILES,required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--actor');p.add_argument('--manifest',required=True);p.add_argument('--name',required=True);p.add_argument('--split');p.add_argument('--entry',choices=['standing','rsi'],default='standing');p.add_argument('--workers',type=int,default=4);p.add_argument('--anchors',action='store_true');a=p.parse_args();out=R/'outputs_amass/jump_tracker_20261007/general_evaluation'/a.name;out.mkdir(parents=True,exist_ok=False);actor=out/'actor.pt'
 if a.actor:
  shutil.copy2(a.actor,actor);shutil.copy2(Path(a.actor).with_suffix('.json'),actor.with_suffix('.json'))
 else:
  with (out/'export.log').open('w') as f:subprocess.run([sys.executable,str(R/'work/unified_export_actor_20261004.py'),'--checkpoint',a.checkpoint,'--output',str(actor)],stdout=f,stderr=subprocess.STDOUT,check=True)

 shared_model=mujoco.MjModel.from_binary_path(str(D/'table_evaluation/baseline_stable/scene_000/reference_input/scene.mjb'))
 for bid in range(shared_model.nbody):
  if shared_model.body(bid).name=='demo_table':shared_model.body_pos[bid,:2]=[100.,100.]
 apply_model_profile(shared_model,a.profile);mujoco.mj_saveModel(shared_model,str(out/'scene.mjb'));shutil.copy2(__file__,out/'evaluate_cpu_robustness.py')
 rows=json.loads(Path(a.manifest).read_text());rows=[r for r in rows if not a.split or r['split']==a.split];results=[]
 (out/'protocol.json').write_text(json.dumps(dict(robustness_profile=a.profile,profile_definition=description(a.profile),checkpoint=a.checkpoint,checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),manifest=a.manifest,manifest_sha256=hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest(),split=a.split,causal_reference_anchoring=a.anchors,planned=len(rows),entry=a.entry,initialization=('1s standing transition; source model floor alignment; zero initial velocity' if a.entry=='standing' else 'Original reference pose and velocity; strip artificial 1s standing prefix'),fall_criterion='pelvis z<0.35m or root tilt>60deg or nonfinite',backend='native CPU; not pooled with historical GPU benchmark'),indent=2))
 with ProcessPoolExecutor(a.workers,initializer=initialize,initargs=(str(actor),a.checkpoint,str(out),a.entry,a.anchors,a.profile)) as pool:
  for r in pool.map(execute,rows):
   results.append(r);(out/'results.json').write_text(json.dumps(results,indent=2));print(len(results),r['task'],r.get('physical_complete'),r.get('error',''),flush=True)
 summary={}
 for task in sorted({r['task'] for r in results}):
  rr=[r for r in results if r['task']==task];complete=[r for r in rr if r.get('physical_complete')];summary[task]=dict(planned=len(rr),complete=len(complete),runtime_errors=sum('error' in r for r in rr),mean_global_mpjpe_completed_m=float(np.mean([r['global_mpjpe_m'] for r in complete])) if complete else None)
 (out/'summary.json').write_text(json.dumps(summary,indent=2))
