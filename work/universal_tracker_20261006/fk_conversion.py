"""Convert uniform GMR candidates with CPU FK; audit against existing GPU-env contract."""
import json,sys,hashlib,os
from pathlib import Path
import numpy as np,mujoco
from scipy.spatial.transform import Rotation,Slerp
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/universal_tracker_20261006'
sys.path[:0]=[str(R/'work'),str(R/'outputs_amass/franken_eleven_20261003/code')]
import transfer as tr
contract=json.loads((D/'evaluation/baseline_broad_dev/contract.json').read_text())
scene=D/'table_evaluation/baseline_stable/scene_000/reference_input/scene.mjb'
model=mujoco.MjModel.from_binary_path(str(scene));data=mujoco.MjData(model)
ids=[model.joint('robot/'+n).id for n in contract['joint_names']]
jq=model.jnt_qposadr[ids];jv=model.jnt_dofadr[ids]
bodies=[model.body('robot/'+n).id for n in contract['body_names']]
root=model.body('robot/pelvis').id
source_model=tr.rt.load_model();names=[source_model.joint(i).name for i in range(1,source_model.njnt)]
order=[names.index(n) for n in contract['joint_names']]
keys=['joint_pos','joint_vel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w']
def convert(ref,entry=True):
 a=np.arange(20)/20.;a=a*a*(3-2*a);lead=np.repeat(ref[:1],20,axis=0);lead[:,7:]=tr.rt.Q0+a[:,None]*(ref[0,7:]-tr.rt.Q0)
 states=np.r_[lead,ref] if entry else ref.copy();t=np.arange(len(states))*.05;tt=np.arange(int(np.ceil(t[-1]/.02))+1)*.02;sample=np.minimum(tt,t[-1])
 pos=np.stack([np.interp(sample,t,c) for c in states[:,:3].T],1);rots=Slerp(t,Rotation.from_quat(states[:,[4,5,6,3]]))(sample);quat=rots.as_quat()[:,[3,0,1,2]]
 q=np.stack([np.interp(sample,t,c) for c in states[:,7:].T],1)[:,order];dq=np.gradient(q,.02,axis=0);vel=np.gradient(pos,.02,axis=0);av=(rots[1:]*rots[:-1].inv()).as_rotvec()/.02;av=np.r_[av,av[-1:]]
 log={k:[] for k in keys}
 for i in range(len(q)):
  data.qpos[:7]=np.r_[pos[i],quat[i]];data.qpos[jq]=q[i];data.qvel[:6]=np.r_[vel[i],rots[i].inv().apply(av[i])];data.qvel[jv]=dq[i];mujoco.mj_forward(model,data)
  xyz=data.xpos[bodies].copy();cv=data.cvel[bodies];ang=cv[:,:3];lin=cv[:,3:]+np.cross(ang,xyz-data.subtree_com[root]);values=[q[i],dq[i],xyz,data.xquat[bodies],lin,ang]
  for k,v in zip(keys,values):log[k].append(v.copy())
 return {k:np.asarray(v,dtype=np.float32) for k,v in log.items()}
