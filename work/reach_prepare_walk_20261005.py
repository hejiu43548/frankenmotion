"""GMR walk with initial and stopping blends only. No reach or scene-aware IK."""
import sys,json,argparse,numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation,Slerp
R=Path('/home/pku/frankenmotion');sys.path.insert(0,str(R/'work'));import unified_retarget_20261004 as rt
p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args();folder=Path(a.folder);rt.init();m=rt.g.M;tr=rt.g.tr;d=tr.mujoco.MjData(m)
def blend(q0,q1,n):
 u=np.linspace(0,1,n+1)[1:];u=u*u*(3-2*u);q=q0[None]+u[:,None]*(q1-q0)[None];q[:,3:7]=Slerp([0,1],Rotation.from_quat(np.array([q0,q1])[:,[4,5,6,3]]))(u).as_quat()[:,[3,0,1,2]];return q
for row in json.loads((folder/'manifest.json').read_text()):
 dest=Path(row['source']);walk=rt.g.convert(np.load(dest/'human_walk.npz'),'uniform');stand=walk[-1].copy();stand[7:]=tr.rt.Q0;stand[3:7]=Rotation.from_euler('z',row['direction_rad']).as_quat()[[3,0,1,2]];d.qpos[:]=stand;tr.rt.floor_align(m,d);stand[2]=d.qpos[2];stop=blend(walk[-1],stand,30);entry=stand.copy();entry[:3]=walk[0,:3];entry[3:7]=walk[0,3:7];entry[7:]=tr.rt.Q0;lead=np.r_[entry[None],blend(entry,walk[0],20)];q=np.r_[lead,walk,stop];n=len(lead);h=n+len(walk);meta=dict(segments=dict(entry=[0,n],walk=[n,h],settle=[h,len(q)],reach_hold=[len(q),len(q)]),reference_frames=len(q),fps=20,scope='Generated walk with uniform GMR; fixed nominal stance entry and stop interpolation only; no arm target or reach IK.');np.savez_compressed(dest/'reference_contact.npz',reference_qpos=q,fps=20.);(dest/'reference_contact.json').write_text(json.dumps(meta,indent=2));print(row['index'],len(q),flush=True)
