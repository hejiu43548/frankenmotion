"""Recompute demo success from saved physical states, controls and exact native model."""
import os,json,argparse,hashlib
os.environ['MUJOCO_GL']='egl'
from pathlib import Path
import numpy as np,mujoco
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();out=Path(a.run);r=json.loads((out/'result.json').read_text());z=np.load(out/'actual.npz');q=z['qpos'];v=z['qvel'];ctrl=z['ctrl'];m=mujoco.MjModel.from_binary_path(str(out/'scene.mjb'));d=mujoco.MjData(m);hand=m.geom('robot/right_hand_collision').id;table=m.geom('table_top').id;pelvis=m.body('robot/pelvis').id;scene=r['scene'];flags=[];force=[];penetrations=[];palms=[];roots=[];max_palmdiff=0.;max_rootdiff=0.;geometry_match=[]
assert np.isfinite(q).all() and np.isfinite(v).all() and np.isfinite(ctrl).all()
for i in range(len(q)):
 d.qpos[:]=q[i];d.qvel[:]=v[i];d.ctrl[:]=ctrl[i];d.qacc_warmstart[:]=0;mujoco.mj_forward(m,d);palms.append(d.geom_xpos[hand].copy());roots.append(d.xpos[pelvis].copy());active=False
 for j in range(d.ncon):
  c=d.contact[j]
  if table in c.geom and hand in c.geom:
   f=np.zeros(6);mujoco.mj_contactForce(m,d,j,f);top=bool(c.pos[2]>scene['table_top']-.012 and abs(c.frame[2])>.8 and d.geom_xpos[hand,2]>scene['table_top']);active|=bool(top and f[0]>.2)
   if top:force.append(float(f[0]));penetrations.append(max(0.,-float(c.dist)))
 flags.append(active)
palms=np.asarray(palms);roots=np.asarray(roots);max_palmdiff=float(np.max(abs(palms-z['palm'])));max_rootdiff=float(np.max(abs(roots-z['root'])));assert max_palmdiff<1e-8 and max_rootdiff<1e-5
longest=current=0
for active in flags:current=current+1 if active else 0;longest=max(longest,current)
root_error=float(np.linalg.norm(roots[-1,:2]-scene['goal_xy']));palm_error=float(np.linalg.norm(palms[-1]-scene['hand_target']));contact_time=sum(flags)*.02;longest_time=longest*.02;complete=bool(r['termination_time'] is None and r['reference_phase_final']>=r['reference_frames']-2 and z['phases'][-1]+1>=r['reference_frames']-2)
success=bool(complete and root_error<.2 and palm_error<.1 and longest_time>=1.)
assert abs(root_error-r['goal_error_m'])<1e-5 and abs(palm_error-r['palm_target_error_m'])<1e-8
assert abs(longest_time-r['longest_hand_contact_s'])<.020001 and success==r['success'],('Contact re-evaluation changed classification',longest_time,r['longest_hand_contact_s'])
# Independent one-control-period CPU integrations diagnose native GPU dynamics parity.
step_errors=[]
for i in np.linspace(0,len(q)-2,min(24,len(q)-1),dtype=int):
 dd=mujoco.MjData(m);dd.qpos[:]=q[i];dd.qvel[:]=v[i];dd.ctrl[:]=ctrl[i+1];mujoco.mj_forward(m,dd)
 for _ in range(round(.02/m.opt.timestep)):mujoco.mj_step(m,dd)
 step_errors.append(dict(index=int(i),root_m=float(np.max(abs(dd.qpos[:3]-q[i+1,:3]))),joint_rad=float(np.max(abs(dd.qpos[7:]-q[i+1,7:])))))
report=dict(success=success,physical_complete=complete,root_error_m=root_error,palm_error_m=palm_error,longest_top_contact_s=longest_time,total_top_contact_s=contact_time,force_N=dict(max=max(force,default=0),median=float(np.median(force)) if force else 0),max_top_penetration_m=max(penetrations,default=0),max_root_step_m=float(np.linalg.norm(np.diff(roots[:,:2],axis=0),axis=-1).max()),min_pelvis_z=float(roots[:,2].min()),raw_palm_error=max_palmdiff,raw_root_error=max_rootdiff,cpu_one_step_max_root_error_m=max(x['root_m'] for x in step_errors),cpu_one_step_max_joint_error_rad=max(x['joint_rad'] for x in step_errors),cpu_one_step_diagnostic=step_errors,contact_sampling_hz=50,scope='Contact means consecutive recorded 50Hz samples, not a proof of uninterrupted contact between samples. Force is recomputed from actual saved qpos/qvel/actuator ctrl in native MuJoCo. One-step CPU integration is a diagnostic; GPU and CPU solvers may differ.',checkpoint_sha256=r['checkpoint_sha256'],files={name:hashlib.sha256((out/name).read_bytes()).hexdigest() for name in ['actual.npz','scene.mjb','result.json']});(out/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k not in ['cpu_one_step_diagnostic','files']},indent=2))
