"""GMR + explicit environment-aware hand IK; no simulated state edits."""
import os,sys,json,argparse
from pathlib import Path
R=Path('/home/pku/frankenmotion');sys.path.insert(0,str(R/'work'))
import unified_retarget_20261004 as rt
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
from scipy.optimize import least_squares
p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--tag',default='reference');p.add_argument('--lift',action='store_true');p.add_argument('--hold-extra',type=int,default=0);p.add_argument('--contact-preload',type=float,default=0.);a=p.parse_args();folder=Path(a.folder);rt.init();m=rt.g.M;tr=rt.g.tr;d=tr.mujoco.MjData(m)
arm=[j for j in range(1,m.njnt) if m.joint(j).name.startswith('right_') and any(s in m.joint(j).name for s in ['shoulder','elbow','wrist'])];adrs=np.array([m.joint(j).qposadr[0] for j in arm]);handbody=m.joint('right_wrist_yaw_joint').bodyid[0];offset=np.array([.11,.01,0.]);limits=m.jnt_range[arm];rows=json.loads((folder/'manifest.json').read_text())
def palm(q):
 d.qpos[:]=q;tr.mujoco.mj_forward(m,d);return d.xpos[handbody]+d.xmat[handbody].reshape(3,3)@offset

def blend(q0,q1,n):
 u=np.linspace(0,1,n+1)[1:];u=u*u*(3-2*u);out=q0[None]+u[:,None]*(q1-q0)[None];out[:,3:7]=Slerp([0,1],Rotation.from_quat(np.array([q0,q1])[:,[4,5,6,3]]))(u).as_quat()[:,[3,0,1,2]];return out
for row in rows:
 dest=Path(row['source']);walk=rt.g.convert(np.load(dest/'human_walk.npz'),'uniform');reach=rt.g.convert(np.load(dest/'human_reach.npz'),'uniform');np.savez_compressed(dest/'raw_retargets.npz',walk=walk,reach=reach)
 # Approach preserves the complete generated/retargeted path. No endpoint snapping.
 stand=walk[-1].copy();stand[7:]=tr.rt.Q0;stand[3:7]=Rotation.from_euler('z',row['direction_rad']).as_quat()[[3,0,1,2]];d.qpos[:]=stand;tr.rt.floor_align(m,d);stand[2]=d.qpos[2]
 stop=blend(walk[-1],stand,30);start=palm(stand).copy();target=np.array(row['hand_target']);target[2]-=a.contact_preload;prev=stand.copy();frames=[];errors=[]
 for k in range(120+a.hold_extra):
  u=min(k/50,1.);smooth=u*u*(3-2*u);want=start+(target-start)*smooth;want[2]+=.09*np.sin(np.pi*u)

  if a.lift:
   lift=start.copy();lift[2]=row['table_top']+.18;above=target.copy();above[2]=row['table_top']+.18
   if k<30:
    v=k/30;v=v*v*(3-2*v);want=start+(lift-start)*v
   elif k<65:
    v=(k-30)/35;v=v*v*(3-2*v);want=lift+(above-lift)*v
   else:
    v=min((k-65)/25,1);v=v*v*(3-2*v);want=above+(target-above)*v
  seed=stand.copy();seed[adrs]=reach[min(k,len(reach)-1),adrs];seed[adrs]=.2*seed[adrs]+.8*prev[adrs]
  def residual(x):
   q=stand.copy();q[adrs]=x;pos=palm(q);rot=d.xmat[handbody].reshape(3,3);yaw=row['direction_rad'];desired=np.array([np.cos(yaw),np.sin(yaw),0.])
   return np.r_[12*(pos-want),.4*(rot[:,0]-desired),.06*(x-seed[adrs]),.12*(x-prev[adrs])]
  fit=least_squares(residual,np.clip(prev[adrs],limits[:,0]+1e-4,limits[:,1]-1e-4),bounds=(limits[:,0]+1e-4,limits[:,1]-1e-4),max_nfev=40)
  q=stand.copy();q[adrs]=fit.x;errors.append(float(np.linalg.norm(palm(q)-want)));frames.append(q);prev=q
 entry=stand.copy();entry[:3]=walk[0,:3];entry[3:7]=walk[0,3:7];entry[7:]=tr.rt.Q0
 lead=np.r_[entry[None],blend(entry,walk[0],20)];states=np.r_[lead,walk,stop,np.array(frames)]
 np.savez_compressed(dest/(a.tag+'.npz'),reference_qpos=states,fps=20.)
 meta=dict(contact_preload_m=a.contact_preload,robot_height=tr.robot_height(m),approach_endpoint=walk[-1,:2].tolist(),approach_error_m=float(np.linalg.norm(walk[-1,:2]-row['goal_xy'])),hand_ik_max_error_m=max(errors),hand_ik_final_error_m=errors[-1],segments=dict(entry=[0,len(lead)],walk=[len(lead),len(lead)+len(walk)],settle=[len(lead)+len(walk),len(lead)+len(walk)+len(stop)],reach_hold=[len(lead)+len(walk)+len(stop),len(states)]),reference_frames=len(states),fps=20,scope='Generated approach unchanged after GMR; explicit smooth standing transition and scene-aware right-arm IK over generated reach. Actual physics is never set to reference after initialization.')
 (dest/(a.tag+'.json')).write_text(json.dumps(meta,indent=2));print(row['index'],meta,flush=True)
