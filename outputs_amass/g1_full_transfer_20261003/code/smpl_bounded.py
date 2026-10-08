"""Bounded source-pose residuals directly through SONIC mode2, no replacement gait."""
import json
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
import full
import smpl_direct as sm
import g1_runtime as rt
from bounded import CASES

def prepare(case,a,b,heading=None):
    original=np.load(full.OUT/'human'/(case+'.npz'))['poses_axisangle'].astype(float).reshape(-1,22,3)
    poses=original.copy();legs=[1,2,4,5,7,8]
    delta=(a-1)*(poses[:,legs]-poses[:,legs].mean(axis=0))
    delta[:,[0,1],0]+=b;delta[:,[4,5],0]-=b
    delta*=np.minimum(1,.35/np.maximum(np.linalg.norm(delta,axis=-1,keepdims=True),1e-12))
    poses[:,legs]+=delta
    geo=(Rotation.from_rotvec(original[:,legs].reshape(-1,3)).inv()*Rotation.from_rotvec(poses[:,legs].reshape(-1,3))).magnitude()
    local,R=sm.fk(poses.reshape(-1,66));R=Rotation.from_euler('z',-R.as_euler('xyz')[0,2])*R
    local=np.r_[np.repeat(local[:1],20,axis=0),local];quat=R.as_quat();quat=np.r_[np.repeat(quat[:1],20,axis=0),quat]
    t=np.arange(len(local))*.05;tt=np.arange(0,t[-1]-1e-8,.02)
    joints=np.stack([np.interp(tt,t,x) for x in local.reshape(len(local),-1).T],axis=1).reshape(-1,24,3)
    quat=Slerp(t,Rotation.from_quat(quat))(tt).as_quat()[:,[3,0,1,2]]
    if heading is not None:
        source_t=np.maximum(tt-1.,0.)
        progress=np.clip((source_t-heading['start'])/max(heading['end']-heading['start'],.05),0,1)
        e=Rotation.from_quat(quat[:,[1,2,3,0]]).as_euler('xyz');e[:,2]=np.deg2rad(heading['degrees'])*progress
        quat=Rotation.from_euler('xyz',e).as_quat()[:,[3,0,1,2]]
    return joints,quat,dict(amplitude=a,bias=b,source_leg_delta_max_rad=float(geo.max()),source_leg_delta_p95_rad=float(np.quantile(geo,.95)),source_upper_delta_max_rad=float(np.max(np.abs(poses[:,[3,6,9,12,13,14,15,16,17,18,19,20,21]]-original[:,[3,6,9,12,13,14,15,16,17,18,19,20,21]]))))

def run(m,p,case,a,b,label,seed=None,cmd=None,heading=None):
    joints,quat,info=prepare(case,a,b,heading);q,_,root=full.native(m,case);p.set_motion(joints,quat)
    info.update(mode=2,command_mps=cmd,note='G1 q fidelity columns refer to old retarget and do not measure mode2 input fidelity; use source rotation delta and geometric audit.')
    r=full.score(m,p,case,label,(q,quat,root),seed,info)
    if heading is not None:
        r['heading_command']=heading;r['turn_success']=bool(r['complete'] and abs(r['turn_deg']-heading['degrees'])<=5)
    if cmd is not None:r['speed_success']=bool(r['complete'] and abs(r['heading_speed_mps']-cmd)<=.05)
    return r

def main():
    m=rt.load_model();p=sm.Policy();rows=[];validation=[]
    full.save('smpl_bounded_protocol.json',dict(cases=CASES,amplitudes=[.7,1.,1.4,1.8,2.2,2.6],biases=[-.08,0,.08],commands=[.25,.4,.55],max_leg_rotation_residual_rad=.35,unchanged=['source time','root orientation','upper body rotations','generator weights','Sonic weights'],note='Per-source simulator calibration; no unseen-prompt claim.'))
    for case in CASES:
        pool=[]
        for a in [.7,1.,1.4,1.8,2.2,2.6]:
            for b in [-.08,0,.08]:
                r=run(m,p,case,a,b,f'smpl_bounded_{case}_a{a:.1f}_b{b:.2f}')
                rows.append(r);pool.append(r);full.save('smpl_bounded_results.json',rows)
        for cmd in [.25,.4,.55]:
            best=min([r for r in pool if r['complete']],key=lambda r:abs(r['heading_speed_mps']-cmd))
            for seed in [4401,4402,4403]:
                r=run(m,p,case,best['amplitude'],best['bias'],f'smpl_bounded_validate_{case}_v{cmd:.2f}_s{seed}',seed,cmd)
                r['calibration_label']=best['label'];validation.append(r);full.save('smpl_bounded_validation.json',validation)
if __name__=='__main__':main()
