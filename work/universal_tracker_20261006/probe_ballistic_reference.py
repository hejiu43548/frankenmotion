"""Exploratory task-agnostic support grounding and ballistic COM reference repair.
No tracker training, no generator claim, no incorporation into frozen main tests.
Preserves joint trajectories up to time resampling, but changes root XYZ/timing.
"""
import sys,json,hashlib
from pathlib import Path
import numpy as np,mujoco
from scipy.spatial.transform import Rotation,Slerp
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/universal_tracker_20261006';sys.path[:0]=[str(R/'work'),str(R/'outputs_amass/franken_eleven_20261003/code')]
import transfer as tr
from fk_conversion import convert
out=D/'ballistic_reference_probe';out.mkdir(exist_ok=False);(out/'references').mkdir();(out/'motions').mkdir()
m=mujoco.MjModel.from_binary_path(str(D/'table_evaluation/baseline_stable/scene_000/reference_input/scene.mjb'));d=mujoco.MjData(m);source=tr.rt.load_model();qa=m.jnt_qposadr[[m.joint('robot/'+source.joint(i).name).id for i in range(1,source.njnt)]];pelvis=m.body('robot/pelvis').id;feet=[i for i in range(m.ngeom) if 'foot' in m.geom(i).name and 'collision' in m.geom(i).name and m.geom_contype[i]];assert all(m.geom_type[i]==mujoco.mjtGeom.mjGEOM_CAPSULE for i in feet),[(m.geom(i).name,int(m.geom_type[i])) for i in feet]
def geometry(q):
 d.qpos[:7]=q[:7];d.qpos[qa]=q[7:];mujoco.mj_forward(m,d)
 return d.subtree_com[pelvis].copy(),float(min(d.geom_xpos[i,2]-m.geom_size[i,0]-abs(d.geom_xmat[i].reshape(3,3)[2,2])*m.geom_size[i,1] for i in feet))
def resample(q,n):
 at=np.linspace(0,len(q)-1,n);r=np.stack([np.interp(at,np.arange(len(q)),v) for v in q.T],1);r[:,3:7]=Slerp(np.arange(len(q)),Rotation.from_quat(q[:,[4,5,6,3]]))(at).as_quat()[:,[3,0,1,2]];return r
def repair(q):
 com,foot=map(np.asarray,zip(*[geometry(x) for x in q]));air=foot>.08;edges=np.diff(np.r_[False,air,False].astype(int));intervals=list(zip(np.flatnonzero(edges==1),np.flatnonzero(edges==-1)));segments=[];cursor=0;pieces=[];notes=[]
 for a,b in intervals:
  if a==0 or b==len(q) or b-a<4:continue
  lo=a-1;hi=b
  if lo<cursor:continue
  grounded=q[cursor:lo].copy();grounded[:,2]-=foot[cursor:lo];pieces.append(grounded)
  first=q[lo].copy();first[2]-=foot[lo];last=q[hi].copy();last[2]-=foot[hi];c0,_=geometry(first);c1,_=geometry(last);peak=max(float(com[lo:hi+1,2].max()),c0[2]+.01,c1[2]+.01);g=-m.opt.gravity[2]
  target_time=np.sqrt(2*(peak-c0[2])/g)+np.sqrt(2*(peak-c1[2])/g);steps=max(4,round(target_time*20));T=steps/20;v0=(c1[2]-c0[2]+.5*g*T*T)/T;flight=resample(q[lo:hi+1],steps+1);times=np.arange(steps+1)/20;target=c0[None]+times[:,None]/T*(c1-c0)[None];target[:,2]=c0[2]+v0*times-.5*g*times**2
  for i in range(len(flight)):
   current,_=geometry(flight[i]);flight[i,:3]+=target[i]-current
  assert np.max(np.abs(geometry(flight[0])[0]-c0))<1e-8 and np.max(np.abs(geometry(flight[-1])[0]-c1))<1e-8
  pieces.append(flight);cursor=hi+1;notes.append(dict(original_start=int(lo),original_end=int(hi),original_duration_s=(hi-lo)/20,new_duration_s=T,original_com_peak_m=peak,new_continuous_com_peak_m=float(c0[2]+v0*v0/(2*g)),root_translations_changed=True,joint_angles_changed_only_by_time_resampling=True))
 remaining=q[cursor:].copy();remaining[:,2]-=foot[cursor:];pieces.append(remaining);repaired=np.concatenate(pieces);return repaired,notes
rows=json.loads((D/'native_eleven_manifest.json').read_text());results=[]
for row in rows:
 original=np.load(row['reference_path'])['reference_qpos'];repaired,notes=repair(original);name=Path(row['path']).stem;ref=out/'references'/(name+'_uniform.npz');motion=out/'motions'/(name+'.npz');np.savez_compressed(ref,reference_qpos=repaired);np.savez_compressed(motion,fps=50.,**convert(repaired));results.append(dict(row,reference_path=str(ref),motion_path=str(motion),original_reference_path=row['reference_path'],reference_repair=notes,reference_frames_before=len(original),reference_frames_after=len(repaired),reference_sha256=hashlib.sha256(ref.read_bytes()).hexdigest()))
 (out/'manifest.json').write_text(json.dumps(results,indent=2));print(len(results),row['task'],len(notes),flush=True)
(out/'protocol.json').write_text(json.dumps(dict(scope=__doc__,classification='Both-foot bottom>8cm for>=4 original20Hz frames, with surrounding support. Ground all other frames by translating rootZ to minimum foot bottom0. Flight COM follows gravity with linear XY; duration from preserved COM apex,quantized50ms. No task label used. No angular-momentum,torque/friction,impact or support-phase acceleration optimization. This is a limited heuristic diagnostic, not a dynamics-feasible retarget solver.',requests=len(results),repaired_flights=sum(len(r['reference_repair']) for r in results)),indent=2))
