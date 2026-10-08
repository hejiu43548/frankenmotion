"""Measure source command response separately from geometric and physical transfer."""
import json
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
import full
import g1_runtime as rt

def unit(a):return a/np.maximum(np.linalg.norm(a,axis=-1,keepdims=True),1e-9)
def local(v,yaw):
    return np.einsum('nij,nkj->nki',Rotation.from_euler('z',-yaw[:,None]).as_matrix(),v)

def main():
    m=rt.load_model();d=mujoco.MjData(m);records=[]
    humanpairs=[(1,4),(4,7),(2,5),(5,8),(16,18),(18,20),(17,19),(19,21)]
    names=[('left_hip_roll_link','left_knee_link'),('left_knee_link','left_ankle_roll_link'),('right_hip_roll_link','right_knee_link'),('right_knee_link','right_ankle_roll_link'),('left_shoulder_roll_link','left_elbow_link'),('left_elbow_link','left_wrist_yaw_link'),('right_shoulder_roll_link','right_elbow_link'),('right_elbow_link','right_wrist_yaw_link')]
    bids=np.array([[m.body(a).id,m.body(b).id] for a,b in names])
    for src in json.loads((full.OUT/'sources.json').read_text()):
        case=src['case'];j=np.load(full.OUT/'human'/(case+'.npz'))['joints_zup_m']
        axis=unit(np.cross(unit(j[:,1]-j[:,2]),unit(j[:,12]-j[:,0])))
        yaw=np.unwrap(np.arctan2(axis[:,1],axis[:,0]));dv=np.diff(j[:,0,:2],axis=0);head=(yaw[:-1]+yaw[1:])/2
        r=dict(case=case,requested=src['requested'],human_heading_speed_mps=float(np.mean(dv[:,0]*np.cos(head)+dv[:,1]*np.sin(head))*20),human_path_speed_mps=float(np.linalg.norm(dv,axis=1).mean()*20),human_turn_deg=float(np.rad2deg(yaw[-1]-yaw[0])))
        modes={mode:mode+'_'+case for mode in ['baseline','smpl']}
        for method in ['bounded','smpl_bounded']:
            for cmd in [.25,.4,.55]:
                label=f'{method}_validate_{case}_v{cmd:.2f}_s4401'
                if (full.OUT/'rollouts'/(label+'.npz')).exists():modes[f'{method}_v{cmd:.2f}']=label
        for mode,label in modes.items():
            z=np.load(full.OUT/'rollouts'/(label+'.npz'));s=z['qpos'][50:];t=np.arange(len(s))*.02
            hi=np.stack([np.interp(t,np.arange(len(j))*.05,x) for x in j.reshape(len(j),-1).T],axis=1).reshape(-1,24,3)
            hy=np.interp(t,np.arange(len(j))*.05,yaw)
            hv=local(np.stack([hi[:,b]-hi[:,a] for a,b in humanpairs],axis=1),hy)
            robot=[]
            for state in s:
                d.qpos[:]=state;mujoco.mj_forward(m,d);robot.append(d.xpos[bids[:,1]]-d.xpos[bids[:,0]])
            ry=np.unwrap(Rotation.from_quat(s[:,[4,5,6,3]]).as_euler('xyz')[:,2])
            rv=local(np.array(robot),ry)
            angles=np.rad2deg(np.arccos(np.clip(np.sum(unit(hv)*unit(rv),axis=-1),-1,1)))
            hands_h=np.stack([hv[:,4]+hv[:,5],hv[:,6]+hv[:,7]],axis=1)
            hands_r=np.stack([rv[:,4]+rv[:,5],rv[:,6]+rv[:,7]],axis=1)
            lenh=np.stack([np.linalg.norm(hv[:,4:6],axis=-1).sum(axis=1),np.linalg.norm(hv[:,6:8],axis=-1).sum(axis=1)],axis=1)
            lenr=np.stack([np.linalg.norm(rv[:,4:6],axis=-1).sum(axis=1),np.linalg.norm(rv[:,6:8],axis=-1).sum(axis=1)],axis=1)
            hands_h/=lenh[:,:,None];hands_r/=lenr[:,:,None]
            metric=json.loads((full.OUT/'rollouts'/(label+'.json')).read_text())
            r[mode]=dict(complete=metric['complete'],heading_speed_mps=metric['heading_speed_mps'],turn_deg=metric['turn_deg'],yaw_tracking_rmse_deg=metric['yaw_tracking_rmse_deg'],terminal_speed_mps=metric['terminal_speed_mps'],leg_direction_mean_deg=float(angles[:,:4].mean()),arm_direction_mean_deg=float(angles[:,4:].mean()),hand_error_arm_lengths=float(np.linalg.norm(hands_h-hands_r,axis=-1).mean()),hand_motion_range_arm_lengths=np.linalg.norm(np.ptp(hands_r,axis=0),axis=-1).tolist(),contact_body_speed_mps=float(np.nanmean(z['contact_body_speed'][50:])))
        records.append(r)
    full.save('source_transfer_audit.json',dict(note='18 clips, six prompt families, one diffusion seed. Directions remove each body yaw independently; position metric normalizes each arm length. Body origins approximate joint centers. Human and G1 speeds are raw physical meters/s, not morphology-rescaled. Contact body speed is not contact-point slip. Mode2 G1 q tracking is not a valid primary body fidelity score.',rows=records))
    print(json.dumps(records,indent=2))
if __name__=='__main__':main()
