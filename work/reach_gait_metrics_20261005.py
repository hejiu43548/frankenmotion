"""Physical gait diagnostics over the walking segment; not a learned naturalness score."""
import json,argparse
from pathlib import Path
import numpy as np,mujoco
from scipy.signal import find_peaks
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--reference-meta');p.add_argument('--segment',choices=['walk','exit'],default='walk');a=p.parse_args();run=Path(a.run);z=np.load(run/'actual.npz');ref=np.load(run/'motion.npz');c=json.loads((run/'inference_contract.json').read_text());res=json.loads((run/'result.json').read_text());meta=json.loads(Path(a.reference_meta or run.parent/'reference_contact.json').read_text());lo,hi=meta['segments'][a.segment];phases=z['phases'][:len(z['qpos'])];mask=(phases>=lo*2.5+20)&(phases<hi*2.5-20);idx=np.flatnonzero(mask);
if len(idx)<=20:
 (run/(a.segment+'_gait_metrics.json')).write_text(json.dumps(dict(available=False,reason='Insufficient recorded frames in segment',segment=a.segment)));raise SystemExit(0)
m=mujoco.MjModel.from_binary_path(str(run/'scene.mjb'));d=mujoco.MjData(m);footnames=['left_ankle_roll_link','right_ankle_roll_link'];bids=[m.body('robot/'+n).id for n in footnames];bodyids=[i for i in range(m.nbody) if m.body(i).name.startswith('robot/')];rids=[bodyids.index(i) for i in bids];gids=[[i for i in range(m.ngeom) if m.geom(i).name.startswith('robot/'+side+'_foot') and 'collision' in m.geom(i).name] for side in ['left','right']];floor=m.geom('terrain').id;assert all(gids)
clearances=[];footpos=[];forces=[];slips=[];root=[];velocity=[]
for i in idx:
 d.qpos[:]=z['qpos'][i];d.qvel[:]=z['qvel'][i];d.ctrl[:]=z['ctrl'][i];mujoco.mj_forward(m,d);footpos.append(d.xpos[bids].copy());root.append(d.xpos[m.body('robot/pelvis').id].copy());clearances.append([min(mujoco.mj_geomDistance(m,d,g,floor,1.,None) for g in foot) for foot in gids]);f=[0.,0.];sv=[[],[]]
 for j in range(d.ncon):
  con=d.contact[j]
  if floor not in con.geom:continue
  for k,foot in enumerate(gids):
   if any(g in foot for g in con.geom):
    force=np.zeros(6);mujoco.mj_contactForce(m,d,j,force);f[k]+=max(0,float(force[0]))
    if force[0]>3:
     jac=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jac,None,con.pos,bids[k]);sv[k].append(float(np.linalg.norm((jac@d.qvel)[:2])))
 forces.append(f);slips.extend([v for vs in sv for v in vs])
clearances=np.asarray(clearances);footpos=np.asarray(footpos);forces=np.asarray(forces);root=np.asarray(root);contact=forces>5.;ph=np.minimum(z['phases'][idx],len(ref['joint_pos'])-1).astype(int);jids=[c['joint_names'].index(n) for n in ['left_knee_joint','right_knee_joint']];qa=[int(m.joint('robot/'+n).qposadr[0]) for n in c['joint_names']];actual=z['qpos'][idx][:,qa];knees=actual[:,jids];want=ref['joint_pos'][ph][:,jids];swing=[]
for k in range(2):
 peaks,_=find_peaks(clearances[:,k],height=.02,prominence=.012,distance=12);swing.append(dict(count=len(peaks),peak_clearance_m=clearances[peaks,k].tolist(),mean_peak_clearance_m=float(clearances[peaks,k].mean()) if len(peaks) else 0.))
report=dict(segment=a.segment,scene_index=res['scene']['index'],n=len(idx),walk_observed_s=len(idx)*.02,success=res['success'],root_error_m=res['goal_error_m'],palm_error_m=res['palm_target_error_m'],knee_std_deg=np.degrees(knees.std(0)).tolist(),reference_knee_std_deg=np.degrees(want.std(0)).tolist(),knee_amplitude_ratio=(knees.std(0)/(want.std(0)+1e-8)).tolist(),knee_rmse_deg=float(np.degrees(np.sqrt(((knees-want)**2).mean()))),feet_position_rmse_m=float(np.sqrt(((footpos-ref['body_pos_w'][ph][:,rids])**2).mean())),double_support_fraction=float(contact.all(-1).mean()),flight_fraction=float((~contact.any(-1)).mean()),contact_fraction=contact.mean(0).tolist(),clearance_p95_m=np.quantile(clearances,.95,axis=0).tolist(),swing=swing,mean_contact_point_slip_m_s=float(np.mean(slips)) if slips else None,scope='Central walking segment excluding 0.4s at both boundaries. Clearance is minimum signed native geom distance from all foot collision shapes to ground; contacts require summed foot force >5N; swing peaks >2cm and prominence >1.2cm; stance slip is velocity at measured contact points >3N. Diagnostics, not a validated naturalness metric.')
(run/(a.segment+'_gait_metrics.json')).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
