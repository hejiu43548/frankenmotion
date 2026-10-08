"""Development-only natural-reference support/flight consistency diagnostic."""
import argparse,json
from pathlib import Path
import numpy as np,mujoco
from scipy.signal import savgol_filter
from scipy.ndimage import binary_erosion
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();root=Path(a.run);results=[]
for row in json.loads((root/'results.json').read_text()):
 if 'error' in row:continue
 run=Path(row['run']);m=mujoco.MjModel.from_binary_path(str(run/'scene.mjb'));d=mujoco.MjData(m);c=json.loads((run/'inference_contract.json').read_text());ref=np.load(run/'motion.npz');qa=m.jnt_qposadr[[m.joint('robot/'+n).id for n in c['joint_names']]];pelvis=m.body('robot/pelvis').id;feet=[i for i in range(m.ngeom) if 'foot' in m.geom(i).name and 'collision' in m.geom(i).name and m.geom_contype[i]];assert feet;com=[];bottoms=[];contacts=[]
 start=50 if row.get('entry','standing')=='standing' else 0
 for i in range(start,len(ref['joint_pos'])):
  d.qpos[:]=m.qpos0;d.qpos[:3]=ref['body_pos_w'][i,0];d.qpos[3:7]=ref['body_quat_w'][i,0];d.qpos[qa]=ref['joint_pos'][i];mujoco.mj_forward(m,d);com.append(d.subtree_com[pelvis].copy());lo=[]
  for gid in feet:
   typ=m.geom_type[gid];rot=d.geom_xmat[gid].reshape(3,3);size=m.geom_size[gid]
   if typ==mujoco.mjtGeom.mjGEOM_BOX:extent=float(np.abs(rot[2])@size)
   elif typ==mujoco.mjtGeom.mjGEOM_SPHERE:extent=float(size[0])
   elif typ==mujoco.mjtGeom.mjGEOM_CAPSULE:extent=float(size[0]+abs(rot[2,2])*size[1])
   else:raise ValueError(int(typ))
   lo.append(float(d.geom_xpos[gid,2]-extent))
  bottoms.append(min(lo));contacts.append(any((m.body_weldid[int(m.geom_bodyid[int(c.geom1)])]==0)!=(m.body_weldid[int(m.geom_bodyid[int(c.geom2)])]==0) for c in d.contact))
 com=np.asarray(com);bottoms=np.asarray(bottoms);air=(bottoms>.02)&~np.asarray(contacts);interior=binary_erosion(air,structure=np.ones(11),border_value=0);acc=savgol_filter(com,11,3,deriv=2,delta=.02,axis=0);error=np.linalg.norm(acc-m.opt.gravity,axis=1);residual=float(np.mean(error[interior])) if interior.any() else None
 results.append(dict(task=row['task'],source=row['source'],physical_complete=row['physical_complete'],frames=len(com),reference_air_fraction=float(air.mean()),reference_interior_air_frames=int(interior.sum()),reference_air_gravity_residual_m_s2=residual,reference_foot_penetration_fraction_2cm=float((bottoms<-.02).mean()),reference_foot_bottom_min_m=float(bottoms.min()),reference_foot_bottom_p95_m=float(np.quantile(bottoms,.95)),scope='Geometric contact and filtered acceleration diagnostic, not a feasibility certificate or proof that this caused rollout failure.'))
summary=dict(requests=len(results),with_interior_air=sum(r['reference_interior_air_frames']>0 for r in results),with_interior_residual_above_2=sum(r['reference_air_gravity_residual_m_s2'] is not None and r['reference_air_gravity_residual_m_s2']>2 for r in results),with_foot_penetration_over_2cm_for_10pct_frames=sum(r['reference_foot_penetration_fraction_2cm']>.1 for r in results))
(root/'natural_support_diagnostic.json').write_text(json.dumps(dict(scope=__doc__,criteria='Feet collision bottom>2cm, no robot/static contact; residual on full11-frame contact-free windows, cubic Savitzky-Golay acceleration,50Hz. Penetration diagnostic is independent. Artificial1s entry omitted.',summary=summary,results=results),indent=2));print(json.dumps(summary),flush=True)
