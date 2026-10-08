"""Official mode-2 observations from native FrankenMotion SMPL rotations."""
import os,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
import full
import g1_runtime as rt
OUT=Path(os.environ['FULL_OUT'])

def fk(poses):
    z=np.load(OUT/'official_human_skeleton.npz');J=z['J'].reshape(-1,3);parents=z['parents'].reshape(-1)
    angles=np.pad(poses,((0,0),(0,99))).reshape(-1,55,3)
    R=Rotation.from_rotvec(angles.reshape(-1,3)).as_matrix().reshape(-1,55,3,3)
    globalR=[];positions=[]
    for i in range(55):
        if i==0:globalR.append(R[:,i]);positions.append(np.repeat(J[i][None],len(poses),axis=0))
        else:
            p=int(parents[i]);globalR.append(globalR[p]@R[:,i]);positions.append(positions[p]+np.einsum('nij,j->ni',globalR[p],J[i]-J[p]))
    joints=np.stack(positions,axis=1)[:,list(range(22))+[39,54]]
    corrected=Rotation.from_rotvec(poses[:,:3])*Rotation.from_quat([.5,.5,.5,.5]).inv()
    local=np.einsum('nij,nkj->nki',corrected.as_matrix().transpose(0,2,1),joints)
    return local,corrected

def prepare(case):
    poses=np.load(OUT/'human'/(case+'.npz'))['poses_axisangle'].astype(float)
    local,R=fk(poses)
    # No extra Y->Z rotation: source native global orientation is already Z-up.
    yaw=R.as_euler('xyz')[0,2];R=Rotation.from_euler('z',-yaw)*R
    # One second first-pose hold. Same clocks, no semantic edits or replacement legs.
    local=np.r_[np.repeat(local[:1],20,axis=0),local]
    quat=R.as_quat();quat=np.r_[np.repeat(quat[:1],20,axis=0),quat]
    t=np.arange(len(local))/20;tt=np.arange(0,t[-1]-1e-8,.02)
    local50=np.stack([np.interp(tt,t,x) for x in local.reshape(len(local),-1).T],axis=1).reshape(-1,24,3)
    quat50=Slerp(t,Rotation.from_quat(quat))(tt).as_quat()[:,[3,0,1,2]]
    return local50,quat50

class Policy(rt.Policy):
    def set_motion(self,joints,quat):self.joints=joints;self.quat=quat
    def act(self,d,q,dq,quat,i):
        ids=np.minimum(i+np.arange(10),len(q)-1)
        refR=Rotation.from_quat(self.quat[ids][:,[1,2,3,0]]).as_matrix()
        robotR=Rotation.from_quat(d.qpos[[4,5,6,3]]).as_matrix()
        enc=np.zeros(1762,dtype=np.float32);enc[0]=2
        enc[922:1642]=self.joints[ids].ravel()
        enc[1642:1702]=(robotR.T@refR)[:,:,:2].ravel()
        enc[1702:1762]=q[ids][:,rt.MJ_TO_IL][:,23:29].ravel()
        token=self.enc.run(None,{'obs_dict':enc[None]})[0].reshape(-1)
        obs=np.concatenate([token]+[np.asarray(h).ravel() for h in self.hist]).astype(np.float32)
        raw=self.dec.run(None,{'obs_dict':obs[None]})[0].reshape(29)
        if not np.isfinite(raw).all():raise ValueError('Nonfinite mode2 action')
        raw=np.clip(raw,-20,20);return rt.Q0+raw[rt.IL_TO_MJ]*rt.SCALE,raw

def main():
    m=rt.load_model();policy=Policy();rows=[]
    full.save('smpl_protocol.json',dict(mode=2,asset='verified official human_joints_info.pkl',native_pose66='global Z-up + 21 local body rotations',joint_indices=list(range(22))+[39,54],offsets={'mode':[0,4],'local_joints':[922,1642],'relative_root_orientation':[1642,1702],'wrists':[1702,1762]},future_frames=10,step_ticks=1,prep='1 second first-pose hold',unchanged_generator=True,note='Primary body input retains native human pose. G1 wrist reference reused. No shared gait. Root translation remains absent from this mode.'))
    for row in json.loads((OUT/'sources.json').read_text()):
        case=row['case'];joints,quat=prepare(case);q,_,root=full.native(m,case);policy.set_motion(joints,quat)
        result=full.score(m,policy,case,'smpl_'+case,(q,quat,root),extra=dict(mode=2,note='q reference metrics are diagnostic vs old G1 retarget, not the mode2 primary human input'))
        rows.append(result);full.save('smpl_results.json',rows)
if __name__=='__main__':main()
