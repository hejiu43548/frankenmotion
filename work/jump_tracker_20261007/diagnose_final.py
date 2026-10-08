"""Reconstruct instantaneous MuJoCo dynamics from recorded states, not a new rollout."""
import os
os.environ['OMP_NUM_THREADS']='1'
import json,argparse,numpy as np,mujoco
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);p.add_argument('--source',default='jump_p0_s0');a=p.parse_args();D=Path(a.output);D.mkdir(parents=True,exist_ok=False)
rows=json.loads((Path(a.run)/'audited_results.json').read_text());rows=[r for r in rows if r['source']==a.source and r['command_set']=='standard_grid'];summary=[]
for row in rows:
 if row['task']!='jump' or 'error' in row:continue
 run=Path(row['run']);m=mujoco.MjModel.from_binary_path(str(run/'scene.mjb'));d=mujoco.MjData(m);z=np.load(run/'actual.npz');ref=np.load(run/'motion.npz');c=json.loads((run/'inference_contract.json').read_text());root=m.body('robot/pelvis').id
 qa=[m.jnt_qposadr[m.joint('robot/'+n).id] for n in c['joint_names']]
 feet=[m.body('robot/'+n).id for n in ['left_ankle_roll_link','right_ankle_roll_link']]
 traces={k:[] for k in ['com','com_velocity','feet_z','force','force_ratio','root_acceleration','reference_com']}
 for i in range(len(z['qpos'])):
  d.qpos[:]=z['qpos'][i];d.qvel[:]=z['qvel'][i];d.ctrl[:]=z['ctrl'][max(0,min(i-1,len(z['ctrl'])-1))];mujoco.mj_forward(m,d);mujoco.mj_subtreeVel(m,d)
  traces['com'].append(d.subtree_com[root].copy());traces['com_velocity'].append(d.subtree_linvel[root].copy());traces['feet_z'].append(d.xpos[feet,2].copy());traces['force'].append(d.actuator_force.copy());traces['force_ratio'].append(abs(d.actuator_force)/np.maximum(abs(m.actuator_forcerange).max(1),1e-6));traces['root_acceleration'].append(d.qacc[:3].copy())
  d.qpos[:7]=np.r_[ref['body_pos_w'][i,0],ref['body_quat_w'][i,0]];d.qpos[qa]=ref['joint_pos'][i];mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);traces['reference_com'].append(d.subtree_com[root].copy())
 traces={k:np.array(v) for k,v in traces.items()};traces['reference_com_velocity']=np.gradient(traces['reference_com'],.02,axis=0)
 np.savez_compressed(D/(run.name+'.npz'),**traces,qpos=z['qpos'],reference_root=ref['body_pos_w'][:,0],fps=50.)
 summary.append(dict(command=row['command'],source=row['source'],actual=row['actual'],g1=row['g1'],peak_vertical_com_velocity=float(traces['com_velocity'][50:,2].max()),reference_peak_vertical_com_velocity=float(traces['reference_com_velocity'][50:,2].max()),peak_motor_force_ratio=float(traces['force_ratio'].max()),saturated_motor_sample_fraction=float((traces['force_ratio']>.98).mean()),mass=float(m.body_subtreemass[root]),actuator_names=[m.actuator(k).name for k in range(m.nu)],per_motor_saturation_fraction=(traces['force_ratio']>.98).mean(0).tolist()))
(D/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps([{k:v for k,v in r.items() if k not in ['actuator_names','per_motor_saturation_fraction','actual','g1']} for r in summary],indent=2))
