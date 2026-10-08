"""Contact-free COM acceleration diagnostic on the same native robot model.
Not a general reference-feasibility certificate; finite-difference/filter dependent.
"""
import argparse,json
from pathlib import Path
import numpy as np,mujoco
from scipy.signal import savgol_filter
from scipy.ndimage import binary_erosion
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();root=Path(a.run);results=[]
def describe(model,states):
 d=mujoco.MjData(model);pelvis=model.body('robot/pelvis').id
 foot=[i for i in range(model.ngeom) if 'foot' in model.geom(i).name and 'collision' in model.geom(i).name and model.geom_contype[i]]
 assert foot
 com=[];floor=[];external_contact=[]
 for q in states:
  d.qpos[:]=q;mujoco.mj_forward(model,d);com.append(d.subtree_com[pelvis].copy());bottoms=[]
  for gid in foot:
   typ=model.geom_type[gid];rot=d.geom_xmat[gid].reshape(3,3);size=model.geom_size[gid]
   if typ==mujoco.mjtGeom.mjGEOM_BOX:extent=float(np.abs(rot[2])@size)
   elif typ==mujoco.mjtGeom.mjGEOM_SPHERE:extent=float(size[0])
   elif typ==mujoco.mjtGeom.mjGEOM_CAPSULE:extent=float(size[0]+abs(rot[2,2])*size[1])
   else:raise ValueError(('Unsupported foot collision geometry',int(typ)))
   bottoms.append(float(d.geom_xpos[gid,2]-extent))
  floor.append(min(bottoms));external_contact.append(any((model.body_weldid[int(model.geom_bodyid[int(c.geom1)])]==0)!=(model.body_weldid[int(model.geom_bodyid[int(c.geom2)])]==0) for c in d.contact))
 com=np.asarray(com);acc=savgol_filter(com,11,3,deriv=2,delta=.02,axis=0);air=(np.asarray(floor)>.02)&~np.asarray(external_contact);interior=binary_erosion(air,structure=np.ones(11),border_value=0);err=np.linalg.norm(acc-model.opt.gravity,axis=1)
 return dict(sample_count=len(states),air_frames=int(air.sum()),interior_air_frames=int(interior.sum()),air_duration_s=float(air.sum()*.02),minimum_foot_height_m=float(min(floor)),maximum_foot_height_m=float(max(floor)),com_vertical_excursion_m=float(np.ptp(com[:,2])),interior_acceleration_residual_mean_m_s2=float(err[interior].mean()) if interior.any() else None,interior_acceleration_residual_p95_m_s2=float(np.quantile(err[interior],.95)) if interior.any() else None,interior_vertical_acceleration_mean_m_s2=float(acc[interior,2].mean()) if interior.any() else None,series=dict(com=com.tolist(),minimum_foot_height_m=floor,acceleration=acc.tolist(),air=air.tolist(),interior_air=interior.tolist()))
for row in json.loads((root/'results.json').read_text()):
 if row['task']!='jump' or 'error' in row:continue
 run=Path(row['run']);m=mujoco.MjModel.from_binary_path(str(run/'scene.mjb'));c=json.loads((run/'inference_contract.json').read_text());ref=np.load(run/'motion.npz');actual=np.load(run/'actual.npz')['qpos'];qa=m.jnt_qposadr[[m.joint('robot/'+n).id for n in c['joint_names']]];q=np.tile(m.qpos0,(len(ref['joint_pos']),1));q[:,:3]=ref['body_pos_w'][:,0];q[:,3:7]=ref['body_quat_w'][:,0];q[:,qa]=ref['joint_pos']
 results.append(dict(source=row['source'],command=row['command'],physical_complete=row['physical_complete'],reference=describe(m,q[50:]),actual=describe(m,actual[50:])))
out=dict(scope=__doc__,definition='Whole-robot subtree COM. Feet collision-shape bottom>2cm and no robot-to-static contact. Acceleration from11-frame cubic Savitzky-Golay second derivative at50Hz; valid frames require full11-frame window in air. In contact-free flight without external force, COM acceleration equals model gravity. Airless clips have null residual, not zero error.',results=results)
(root/'aerial_dynamics.json').write_text(json.dumps(out,indent=2));print(json.dumps(out,indent=2))
