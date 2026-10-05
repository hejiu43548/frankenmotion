"""Experimental contact-aware retarget refinement. Root XY is never changed."""
from pathlib import Path
import json,shutil,numpy as np,mujoco
from scipy.spatial.transform import Rotation,Slerp
from scipy.optimize import least_squares
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/gait_demo_20261005';out=D/'dev_contact';out.mkdir(exist_ok=False);m=mujoco.MjModel.from_binary_path(str(D/'dev_timed/scene_000/baseline_timed/scene.mjb'));d=mujoco.MjData(m);floor=m.geom('terrain').id;feet=[m.body('robot/'+side+'_ankle_roll_link').id for side in ['left','right']];gids=[[g for g in range(m.ngeom) if m.geom(g).name.startswith('robot/'+side+'_foot') and 'collision' in m.geom(g).name] for side in ['left','right']];legs=[[j for j in range(1,m.njnt) if m.joint(j).name.startswith('robot/'+side+'_') and any(x in m.joint(j).name for x in ['hip_','knee_','ankle_'])] for side in ['left','right']]
def segments(flag):
 bounds=np.r_[0,np.flatnonzero(flag[1:]!=flag[:-1])+1,len(flag)];return [(int(a),int(b),bool(flag[a])) for a,b in zip(bounds[:-1],bounds[1:])]
def blend(a,b,n):
 u=np.linspace(0,1,n);u=u*u*(3-2*u);q=a[None]+u[:,None]*(b-a)[None];q[:,3:7]=Slerp([0,1],Rotation.from_quat(np.array([a,b])[:,[4,5,6,3]]))(u).as_quat()[:,[3,0,1,2]];return q
records=[]
for row in json.loads((D/'dev_timed/manifest.json').read_text()):
 src=Path(row['source']);dest=out/src.name;dest.mkdir();r=dict(row);r['source']=str(dest);q=np.load(src/'reference_contact.npz')['reference_qpos'];meta=json.loads((src/'reference_contact.json').read_text());lo,hi=meta['segments']['walk'];walk=q[lo:hi].copy();pos=[];rot=[];clear=[]
 for state in walk:
  d.qpos[:]=state;mujoco.mj_forward(m,d);pos.append(d.xpos[feet].copy());rot.append(d.xquat[feet].copy());clear.append([min(mujoco.mj_geomDistance(m,d,g,floor,1.,None) for g in foot) for foot in gids])
 pos=np.asarray(pos);rot=np.asarray(rot);clear=np.asarray(clear);target=pos.copy();trot=rot.copy();flags=[]
 for f in range(2):
  stance=clear[:,f]<.01;stance[:3]=True;stance[-3:]=True
  for a,b,v in segments(stance):
   if b-a<3 and a>0 and b<len(stance):stance[a:b]=not v
  segs=segments(stance)
  for a,b,v in segs:
   if v:
    anchor=pos[(a+b-1)//2,f].copy();anchor[2]=.035;target[a:b,f]=anchor;yaw=Rotation.from_quat(rot[(a+b-1)//2,f,[1,2,3,0]]).as_euler('xyz')[2];trot[a:b,f]=Rotation.from_euler('z',yaw).as_quat()[[3,0,1,2]]
  for a,b,v in segs:
   if not v:
    assert a>0 and b<len(walk);u=np.arange(1,b-a+1)/(b-a+1);s=u*u*(3-2*u);target[a:b,f]=target[a-1,f]+s[:,None]*(target[b,f]-target[a-1,f]);target[a:b,f,2]+=.045*np.sin(np.pi*u);trot[a:b,f]=Slerp([0,1],Rotation.from_quat(trot[[a-1,b],f][:,[1,2,3,0]]))(s).as_quat()[:,[3,0,1,2]]
  flags.append(stance)
 result=walk.copy();errors=[]
 for i in range(len(walk)):
  state=walk[i].copy()
  for f in range(2):
   js=legs[f];qa=np.array([int(m.jnt_qposadr[j]) for j in js]);lim=m.jnt_range[js];want=Rotation.from_quat(trot[i,f,[1,2,3,0]]);seed=walk[i,qa];previous=result[max(0,i-1),qa]
   def residual(x):
    d.qpos[:]=state;d.qpos[qa]=x;mujoco.mj_kinematics(m,d);rr=Rotation.from_quat(d.xquat[feet[f]][[1,2,3,0]]);return np.r_[30*(d.xpos[feet[f]]-target[i,f]),2*(want.inv()*rr).as_rotvec(),.05*(x-seed),.025*(x-previous)]
   fit=least_squares(residual,np.clip(previous,lim[:,0]+1e-4,lim[:,1]-1e-4),bounds=(lim[:,0]+1e-4,lim[:,1]-1e-4),max_nfev=35);state[qa]=fit.x;residual(fit.x);errors.append(float(np.linalg.norm(d.xpos[feet[f]]-target[i,f])))
  result[i]=state
 assert np.array_equal(result[:,:7],walk[:,:7]);q[lo:hi]=result;q[:lo]=blend(q[0],result[0],lo);sl,sh=meta['segments']['settle'];q[sl:sh]=blend(result[-1],q[sh-1],sh-sl);meta['contact_refinement']=dict(root_pose_unchanged=True,stance_detection='original foot minimum clearance <1cm, minimum 3-frame segments',stance='fixed midpoint anchor, flat sole',swing='smooth endpoints from generated step timing, 4.5cm lift, interpolated yaw',mean_ik_error_m=float(np.mean(errors)),max_ik_error_m=max(errors),scope='Explicit contact-aware leg IK, not a new learned generator')
 np.savez_compressed(dest/'reference_contact.npz',reference_qpos=q,fps=20.);np.savez_compressed(dest/'contact_plan.npz',feet_target=target,contact=np.array(flags).T,original_clearance=clear);(dest/'reference_contact.json').write_text(json.dumps(meta,indent=2));(dest/'scene.json').write_text(json.dumps(r,indent=2));records.append(r);print(row['index'],meta['contact_refinement'],flush=True)
(out/'manifest.json').write_text(json.dumps(records,indent=2));probe=D/'probe_contact';probe.mkdir();(probe/'manifest.json').write_text(json.dumps(records[:2],indent=2))
