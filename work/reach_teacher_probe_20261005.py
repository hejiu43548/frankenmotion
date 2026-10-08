"""Standalone closed-loop MuJoCo CPU execution; no mjlab/Isaac/CUDA required.
Consumes exported actor, native scene and 50 Hz generated motion reference.
"""
import os,json,argparse,shutil,hashlib
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
import numpy as np,mujoco,torch
from scipy.spatial.transform import Rotation
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--actor',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);p.add_argument('--check-observations',action='store_true');a=p.parse_args();run=Path(a.run);out=Path(a.output);out.mkdir(parents=True,exist_ok=False);c=json.loads((run/'inference_contract.json').read_text());m=mujoco.MjModel.from_binary_path(str(run/'scene.mjb'));d=mujoco.MjData(m);ref=dict(np.load(run/'motion.npz'));actor=torch.jit.load(a.actor,map_location='cpu').eval();torch.set_num_threads(2);anchor=m.body(c['anchor_body_name']).id;jids=[m.joint('robot/'+n).id for n in c['joint_names']];qa=m.jnt_qposadr[jids];va=m.jnt_dofadr[jids];default=np.asarray(c['default_joint_pos']);ridx=c['reference_anchor_index'];aids=[]
for n in c['action_target_names']:
 j=m.joint('robot/'+n).id;ids=np.where(m.actuator_trnid[:,0]==j)[0];assert len(ids)==1;aids.append(int(ids[0]))
assert not np.any(c['encoder_bias']),'This portable runtime currently supports nominal zero encoder bias only'
def sensor(name):
 s=m.sensor(name);return d.sensordata[s.adr[0]:s.adr[0]+s.dim[0]].copy()
def quatrot(wxyz):return Rotation.from_quat(np.asarray(wxyz)[[1,2,3,0]])
def observation(phase,last):
 actual_rotation=quatrot(d.xquat[anchor]);inv=actual_rotation.inv()
 def relative(i):
  pos=inv.apply(ref['body_pos_w'][i,ridx]-d.xpos[anchor]);rot=(inv*quatrot(ref['body_quat_w'][i,ridx])).as_matrix()[:,:2].reshape(-1);return pos,rot
 pos,rot=relative(phase);parts=[ref['joint_pos'][phase],ref['joint_vel'][phase],pos,rot,sensor(c['linear_velocity_sensor']),sensor(c['angular_velocity_sensor']),d.qpos[qa]-default,d.qvel[va],last]
 for offset in c['preview_offsets']:
  i=min(phase+offset,len(ref['joint_pos'])-1);rp,ro=relative(i);parts.extend([ref['joint_pos'][i],ref['joint_vel'][i],rp,ro])
 return np.concatenate(parts).astype(np.float32)
if a.check_observations:
 recorded=np.load(run/'actual.npz');errors=[];action_errors=[];difference=[]
 for i in np.linspace(0,len(recorded['qpos'])-1,30,dtype=int):
  d.qpos[:]=c['initial_qpos'] if i==0 else recorded['qpos'][i-1];d.qvel[:]=c['initial_qvel'] if i==0 else recorded['qvel'][i-1];d.ctrl[:]=0 if i==0 else recorded['ctrl'][i-1];mujoco.mj_forward(m,d);last=np.zeros(29) if i==0 else recorded['actions'][i-1];obs=observation(int(recorded['phases'][i]),last);errors.append(float(np.max(abs(obs-recorded['observations'][i]))));difference.append((obs-recorded['observations'][i]).copy())
  with torch.inference_mode():act=actor(torch.from_numpy(obs)[None])[0].numpy()
  action_errors.append(float(np.max(abs(act-recorded['actions'][i]))))
 bounds=[0,58,61,67,70,73,102,131,160,361];group_errors=[float(np.max(abs(np.asarray(difference)[:,left:right]))) for left,right in zip(bounds[:-1],bounds[1:])]
 report=dict(per_sample_errors=errors,group_errors=group_errors,observation_max_error=max(errors),action_max_error=max(action_errors),samples=len(errors));(out/'observation_parity.json').write_text(json.dumps(report,indent=2));print(report,flush=True)
 assert max(errors)<2e-4 and max(action_errors)<2e-4,report
# Fresh simulation initialized once; all subsequent qpos comes from mj_step.
d=mujoco.MjData(m);d.qpos[:]=c['initial_qpos'];d.qvel[:]=c['initial_qvel'];mujoco.mj_forward(m,d);last=np.zeros(29);states=[];velocities=[];controls=[];acts=[];obslog=[];phases=[];roots=[];palms=[];contacts=[];hand=m.geom('robot/right_hand_collision').id;pelvis=m.body('robot/pelvis').id;table=m.geom('table_top').id;source_result=json.loads((run/'result.json').read_text());source_result['checkpoint']=a.checkpoint;source_result['checkpoint_sha256']=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest();actor_metadata=json.loads(Path(a.actor).with_suffix('.json').read_text());assert actor_metadata['checkpoint_sha256']==source_result['checkpoint_sha256'];source_result['actor_sha256']=hashlib.sha256(Path(a.actor).read_bytes()).hexdigest();scene=source_result['scene'];termination=None
with torch.inference_mode():
 for i in range(len(ref['joint_pos'])-2):
  obs=observation(i,last);act=actor(torch.from_numpy(obs)[None])[0].numpy(); j=np.arange(22,29); look=min(i+2,len(ref['joint_pos'])-1); kp=m.actuator_gainprm[np.asarray(aids)[j],0]; kd=-m.actuator_biasprm[np.asarray(aids)[j],2]; target=ref['joint_pos'][look,j]+kd/kp*ref['joint_vel'][look,j]+d.qfrc_bias[np.asarray(va)[j]]/kp; act[j]=(target-np.asarray(c['action_offset'])[j])/np.asarray(c['action_scale'])[j];last=act.copy()
  if c['clip_actions'] is not None:last=np.clip(last,-c['clip_actions'],c['clip_actions'])
  d.ctrl[aids]=last*np.asarray(c['action_scale'])+np.asarray(c['action_offset'])
  for _ in range(round(c['control_timestep']/m.opt.timestep)):mujoco.mj_step(m,d)
  mujoco.mj_forward(m,d);states.append(d.qpos.copy());velocities.append(d.qvel.copy());controls.append(d.ctrl.copy());acts.append(act.copy());obslog.append(obs);phases.append(i);roots.append(d.xpos[pelvis].copy());palms.append(d.geom_xpos[hand].copy())
  found=[]
  for j in range(d.ncon):
   con=d.contact[j]
   if hand in con.geom and table in con.geom:
    force=np.zeros(6);mujoco.mj_contactForce(m,d,j,force);top=bool(con.pos[2]>scene['table_top']-.012 and abs(con.frame[2])>.8 and d.geom_xpos[hand,2]>scene['table_top']);found.append(dict(distance=float(con.dist),normal_force=float(force[0]),position=con.pos.tolist(),top_surface=top))
  contacts.append(found)
  w,x,y,zz=d.qpos[3:7];roll=np.arctan2(2*(w*x+y*zz),1-2*(x*x+y*y));pitch=np.arcsin(np.clip(2*(w*y-zz*x),-1,1))
  if d.xpos[pelvis,2]<.35 or np.hypot(roll,pitch)>1.05:termination=(i+1)*.02;break
np.savez_compressed(out/'actual.npz',qpos=states,qvel=velocities,ctrl=controls,actions=acts,observations=obslog,phases=phases,root=roots,palm=palms,fps=50.);(out/'execution.json').write_text(json.dumps(dict(backend='DEVELOPMENT ONLY: actor lower body plus reference PD right arm with gravity compensation; NOT deployment policy',source_run=str(run),actor=a.actor,frames=len(states),termination_time=termination,final_root=roots[-1].tolist(),final_palm=palms[-1].tolist(),no_state_writes_after_initialization=True),indent=2));print('CPU execution complete',out,flush=True)

for name in ['scene.mjb','motion.npz','inference_contract.json']:shutil.copy2(run/name,out/name)
flags=[any(x['top_surface'] and x['normal_force']>.2 for x in row) for row in contacts];longest=current=0
for flag in flags:current=current+1 if flag else 0;longest=max(longest,current)
result=dict(source_result);result.update(backend='development_only_analytic_arm_teacher',physical_complete=termination is None and len(states)==len(ref['joint_pos'])-2,termination_time=termination,duration_s=len(states)*.02,reference_phase_final=len(states),root_final=roots[-1].tolist(),palm_final=palms[-1].tolist(),goal_error_m=float(np.linalg.norm(roots[-1][:2]-scene['goal_xy'])),palm_target_error_m=float(np.linalg.norm(palms[-1]-scene['hand_target'])),hand_contact_s=sum(flags)*.02,longest_hand_contact_s=longest*.02)
result['success']=bool(result['physical_complete'] and result['goal_error_m']<.2 and result['palm_target_error_m']<.1 and result['longest_hand_contact_s']>=1.)
(out/'result.json').write_text(json.dumps(result,indent=2));(out/'contacts.json').write_text(json.dumps(contacts));print('CPU success',result['success'],'palm error',result['palm_target_error_m'],flush=True)

import subprocess,sys
subprocess.run([sys.executable,str(Path(__file__).with_name("reach_metrics_20261005.py")),"--run",str(out)],check=True)
result["command_metrics"]=json.loads((out/"command_metrics.json").read_text());result["success"]=result["command_metrics"]["success"];(out/"result.json").write_text(json.dumps(result,indent=2))
