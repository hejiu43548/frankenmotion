"""Fixed-policy, task-agnostic support correction probe; excluded from final pipeline."""
import json,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np,mujoco
from scipy.signal import savgol_filter
from scipy.ndimage import binary_erosion,label,gaussian_filter1d
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/universal_tracker_20261006';source=D/'general_evaluation/natural_v4_3000';out=D/'natural_support_projection'
def prepare(row):
 run=Path(row['run']);m=mujoco.MjModel.from_binary_path(str(run/'scene.mjb'));d=mujoco.MjData(m);c=json.loads((run/'inference_contract.json').read_text());ref=dict(np.load(run/'motion.npz'));qa=m.jnt_qposadr[[m.joint('robot/'+n).id for n in c['joint_names']]];pelvis=m.body('robot/pelvis').id;feet=[i for i in range(m.ngeom) if 'foot' in m.geom(i).name and 'collision' in m.geom(i).name and m.geom_contype[i]];com=[];bottoms=[];contacts=[]
 for i in range(len(ref['joint_pos'])):
  d.qpos[:]=m.qpos0;d.qpos[:3]=ref['body_pos_w'][i,0];d.qpos[3:7]=ref['body_quat_w'][i,0];d.qpos[qa]=ref['joint_pos'][i];mujoco.mj_forward(m,d);com.append(d.subtree_com[pelvis].copy());lo=[]
  for gid in feet:
   typ=m.geom_type[gid];rot=d.geom_xmat[gid].reshape(3,3);size=m.geom_size[gid]
   if typ==mujoco.mjtGeom.mjGEOM_BOX:extent=float(np.abs(rot[2])@size)
   elif typ==mujoco.mjtGeom.mjGEOM_SPHERE:extent=float(size[0])
   elif typ==mujoco.mjtGeom.mjGEOM_CAPSULE:extent=float(size[0]+abs(rot[2,2])*size[1])
   else:raise ValueError(int(typ))
   lo.append(float(d.geom_xpos[gid,2]-extent))
  bottoms.append(min(lo));contacts.append(any((m.body_weldid[int(m.geom_bodyid[int(c.geom1)])]==0)!=(m.body_weldid[int(m.geom_bodyid[int(c.geom2)])]==0) for c in d.contact))
 com=np.asarray(com);bottoms=np.asarray(bottoms);air=(bottoms>.02)&~np.asarray(contacts);air[:50]=False;interior=binary_erosion(air,structure=np.ones(11),border_value=0);acc=savgol_filter(com,11,3,deriv=2,delta=.02,axis=0);error=np.linalg.norm(acc-m.opt.gravity,axis=1);segments,n=label(air);mask=bottoms<-.02
 for k in range(1,n+1):
  core=(segments==k)&interior
  if core.any() and np.median(error[core])>2:mask|=segments==k
 raw=np.where(mask,np.clip(-bottoms,-.12,.12),0.);raw[:50]=0;delta=gaussian_filter1d(raw,5);fade=np.clip((np.arange(len(delta))-50)/25,0,1);delta*=fade*fade*(3-2*fade);modified={k:v.copy() for k,v in ref.items()};modified['body_pos_w'][:,:,2]+=delta[:,None];modified['body_lin_vel_w'][:,:,2]+=np.gradient(delta,.02)[:,None]
 for k in ['joint_pos','joint_vel','body_quat_w','body_ang_vel_w']:assert np.array_equal(ref[k],modified[k])
 assert np.array_equal(ref['body_pos_w'][:,:,:2],modified['body_pos_w'][:,:,:2]);assert len(ref['joint_pos'])==len(modified['joint_pos'])
 dest=out/(Path(row['path']).stem+'_motion.npz');np.savez_compressed(dest,**modified);record={k:v for k,v in row.items() if k in ['task','uid','source','seed','command','path','reference_path','split','caption','source_type']};record.update(motion_path=str(dest),original_motion_path=row['motion_path'],support_projection_max_abs_m=float(abs(delta).max()),support_projection_frames=int((abs(delta)>.001).sum()),motion_sha256=hashlib.sha256(dest.read_bytes()).hexdigest());return record
if __name__=='__main__':
 out.mkdir(exist_ok=False);rows=json.loads((source/'results.json').read_text());assert len(rows)==57 and all('error' not in r for r in rows)
 protocol=dict(scope=__doc__,source=str(source),checkpoint=str(D/'training/joint_v4_long_physical/model_3000.pt'),method='For contact-free segments with feet>2cm and median full11-frame gravity residual>2m/s2, or foot penetration>2cm, shift global bodyZ towards lowest foot ground. Cap correction at12cm, Gaussian sigma5frames, fade overfirst0.5s afterstanding entry. Joint states, orientation,rootXY and timing unchanged. No angular-momentum/friction/takeoff optimization.',selection='All57 historical natural validation clips, all failures retained. Fixed v4 policy. Diagnostic only, no effect on global checkpoint selection or frozen final references.')
 (out/'protocol.json').write_text(json.dumps(protocol,indent=2))
 with ProcessPoolExecutor(4) as pool:records=list(pool.map(prepare,rows))
 (out/'manifest.json').write_text(json.dumps(records,indent=2));print('Prepared',len(records),'modified',sum(r['support_projection_frames']>0 for r in records),flush=True)
