"""Source-preserving migration tests; no shared gait or periodic replacement."""
import os,json,time,argparse
from pathlib import Path
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation,Slerp
from scipy.ndimage import median_filter,gaussian_filter1d
import g1_runtime as rt
import original_experiment as ex
import prior_run as prior
OUT=Path(os.environ['FULL_OUT']);prior.OUT=OUT;ex.OUT=OUT;ex.SOURCES=OUT/'human'
SOURCES=OUT/'retarget'

def save(name,value):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);tmp=p.with_suffix(p.suffix+'.tmp');tmp.write_text(json.dumps(value,indent=2));tmp.replace(p)

def native(m,case):return ex.reference(m,SOURCES/(case+'.npz'),np.zeros(7))

def body_fk(m,q,quat,root):
    d=mujoco.MjData(m);ps=[];rs=[]
    bids=[m.body(s+'_ankle_roll_link').id for s in ['left','right']]
    for qi,ori,p in zip(q,quat,root):
        d.qpos[:]=np.r_[p,ori,qi];mujoco.mj_forward(m,d)
        ps.append(d.xpos[bids].copy());rs.append(d.xmat[bids].copy().reshape(2,3,3))
    return np.array(ps),np.array(rs)

def contact_reference(m,case,speed,strength,command_heading=False):
    # Work at native 20Hz, preserve every source frame and its inferred foot phase.
    z=np.load(SOURCES/(case+'.npz'));q=z['q'].copy();root=z['root'].copy();quat=z['quat'].copy();n=len(q)
    foot,Rfoot=body_fk(m,q,quat,root)
    rel=foot-root[:,None,:]
    # Reference-derived support, no prescribed gait frequency or replacement cycle.
    height=foot[:,:,2]-np.min(foot[:,:,2],axis=1,keepdims=True)
    support=median_filter((height<.018).astype(float),size=(3,1))>.5
    # Ensure at least one estimated support for walking sources; jogging may be aerial.
    for t in range(n):
        if not support[t].any():support[t,np.argmin(height[t])]=True
    manifest={r['case']:r for r in json.loads((OUT/'sources.json').read_text())};request=manifest[case]['requested']
    yaw=np.unwrap(Rotation.from_quat(quat[:,[1,2,3,0]]).as_euler('xyz')[:,2])
    if command_heading:
        begin=request.get('turn_start',0);end=request.get('turn_end',(n-1)/20)
        progress=np.clip((np.arange(n)/20-begin)/max(end-begin,.05),0,1)
        yaw=np.deg2rad(request['turn'])*progress
        quat=Rotation.from_euler('z',yaw).as_quat()[:,[3,0,1,2]]
    # Desired translation induces stance-foot relative velocity, making speed visible in q/dq.
    root_new=root.copy();root_new[0,:2]=root[0,:2]
    stop=request.get('stop')
    for t in range(1,n):
        v=0. if stop is not None and t/20>=float(stop) else speed
        root_new[t,:2]=root_new[t-1,:2]+v*.05*np.array([np.cos(yaw[t]),np.sin(yaw[t])])
    target=root_new[:,None,:]+rel
    anchors=np.zeros((2,3));original_anchor=np.zeros((2,3))
    for t in range(n):
        for k in range(2):
            if support[t,k]:
                if t==0 or not support[t-1,k]:anchors[k]=target[t,k];original_anchor[k]=foot[t,k]
                # Hard world-space contact anchor blended with original relative foot shape.
                target[t,k]=(1-strength)*target[t,k]+strength*anchors[k]
    result=[];residuals=[];prev=q[0,:12].copy();d=mujoco.MjData(m)
    feet=[m.body(s+'_ankle_roll_link').id for s in ['left','right']]
    bounds=(m.jnt_range[1:13,0]+1e-4,m.jnt_range[1:13,1]-1e-4)
    for t in range(n):
        d.qpos[:]=np.r_[root_new[t],quat[t],q[t]]
        def loss(legs):
            d.qpos[7:19]=legs;mujoco.mj_forward(m,d)
            p=d.xpos[feet];r=d.xmat[feet].reshape(2,3,3)
            return np.r_[((p-target[t])*24).ravel(),((r-Rfoot[t])* .3).ravel(),.65*(legs-q[t,:12]),.15*(legs-prev)]
        fit=least_squares(loss,np.clip(prev,*bounds),bounds=bounds,max_nfev=24,ftol=2e-4,xtol=2e-4)
        current=q[t].copy();current[:12]=fit.x;result.append(current);prev=fit.x
        residuals.append(float(np.linalg.norm(loss(fit.x)[:6])/24))
    result=np.asarray(result)
    delta=result-q
    foot_new,_=body_fk(m,result,quat,root_new)
    vel=np.linalg.norm(np.diff(foot_new,axis=0)/.05,axis=-1)
    contact_speed=float(np.mean(vel[support[1:]&support[:-1]]))
    info=dict(case=case,speed=speed,strength=strength,command_heading=command_heading,
        lower_delta_rms_rad=float(np.sqrt(np.mean(delta[:,:12]**2))),lower_delta_p95_rad=float(np.quantile(np.abs(delta[:,:12]),.95)),
        upper_delta_max_rad=float(np.max(np.abs(delta[:,12:]))),ik_foot_error_mean_m=float(np.mean(residuals)),
        estimated_support_fraction=support.mean(axis=0).tolist(),estimated_stance_foot_speed_mps=contact_speed,
        note='Contact inferred from reference foot heights; not ground-truth labels. No source frame deletion, retiming or template gait.')
    return result,quat,root_new,info,support

def upsample(q,quat,root):
    a=np.linspace(0,1,21)[:-1];a=a*a*(3-2*a)
    q=np.r_[rt.Q0+a[:,None]*(q[0]-rt.Q0),q]
    root=np.r_[np.repeat(root[:1],20,axis=0),root];quat=np.r_[np.repeat(quat[:1],20,axis=0),quat]
    t=np.arange(len(q))/20;tt=np.arange(0,t[-1]-1e-8,.02)
    qi=np.stack([np.interp(tt,t,q[:,k]) for k in range(29)],axis=1)
    ri=np.stack([np.interp(tt,t,root[:,k]) for k in range(3)],axis=1)
    qu=Slerp(t,Rotation.from_quat(quat[:,[1,2,3,0]]))(tt).as_quat()[:,[3,0,1,2]]
    return qi,qu,ri

def score(m,policy,case,label,ref,seed=None,extra=None):
    q,quat,root=ref;start=time.monotonic();arrays,fall=prior.rollout(m,policy,q,quat,root,seed)
    s=arrays['qpos'];complete=fall is None and len(s)==len(q)
    baseline,_,_=native(m,case);err=s[:,7:]-q[:len(s)]
    r=dict(case=case,label=label,seed=seed,complete=complete,fall_time_s=fall,wall_s=time.monotonic()-start,
       lower_tracking_rmse_rad=float(np.sqrt(np.mean(err[50:,:12]**2))) if len(s)>50 else None,
       upper_tracking_rmse_rad=float(np.sqrt(np.mean(err[50:,12:]**2))) if len(s)>50 else None,
       reference_lower_delta_rms_rad=float(np.sqrt(np.mean((q-baseline)[:,:12]**2))),
       reference_lower_delta_p95_rad=float(np.quantile(np.abs((q-baseline)[:,:12]),.95)),
       reference_upper_delta_max_rad=float(np.max(np.abs((q-baseline)[:,12:]))),
       joint_speed_max_rad_s=float(np.max(np.abs(np.gradient(q,.02,axis=0)))))
    if complete:
        yaw=np.unwrap(Rotation.from_quat(s[:,[4,5,6,3]]).as_euler('xyz')[:,2]);head=(yaw[50:-1]+yaw[51:])/2;delta=np.diff(s[50:,:2],axis=0)
        refyaw=np.unwrap(Rotation.from_quat(quat[:,[1,2,3,0]]).as_euler('xyz')[:,2])
        r.update(speed_mps=float((s[-1,0]-s[50,0])/((len(s)-51)*.02)),heading_speed_mps=float(np.mean(delta[:,0]*np.cos(head)+delta[:,1]*np.sin(head))/.02),
          turn_deg=float(np.rad2deg(yaw[-1]-yaw[50])),reference_turn_deg=float(np.rad2deg(refyaw[-1]-refyaw[50])),
          yaw_tracking_rmse_deg=float(np.rad2deg(np.sqrt(np.mean((yaw[50:]-refyaw[50:])**2)))),
          max_heading_drift_deg=float(np.max(np.abs(yaw[50:]-yaw[50]))*180/np.pi),
          terminal_speed_mps=float(np.mean(np.linalg.norm(arrays['qvel'][-25:,:2],axis=1))))
    if extra:r.update(extra)
    np.savez_compressed(OUT/'rollouts'/(label+'.npz'),**arrays,reference_qpos=np.c_[root,quat,q],fps=50.)
    save('rollouts/'+label+'.json',r)
    with (OUT/'ledger.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
    print(json.dumps(r),flush=True);return r

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['retarget','baseline','contact'],required=True);args=ap.parse_args()
    (OUT/'rollouts').mkdir(parents=True,exist_ok=True);m=rt.load_model()
    if args.stage=='retarget':ex.retarget(m);return
    policy=rt.Policy();rows=[]
    cases=[r['case'] for r in json.loads((OUT/'sources.json').read_text())]
    if args.stage=='baseline':
        for case in cases:rows.append(score(m,policy,case,'baseline_'+case,native(m,case)))
    else:
        cases=['01_walk_speed_v1','01_walk_speed_v3','04_walk_wave_v1','04_walk_wave_v3']
        save('contact_protocol.json',dict(cases=cases,speeds=[.25,.4,.55],strengths=[.35,.7,1.],fidelity_gate='lower delta P95 <=0.35 rad, upper unchanged, no retiming/no gait replacement',speed_tolerance_mps=.05))
        for case in cases:
            for v in [.25,.4,.55]:
                for strength in [.35,.7,1.]:
                    label=f'contact_{case}_v{v:.2f}_s{strength:.2f}'
                    q,quat,root,info,support=contact_reference(m,case,v,strength)
                    ref=upsample(q,quat,root)
                    (OUT/'references').mkdir(exist_ok=True)
                    np.savez_compressed(OUT/'references'/(label+'.npz'),q=q,quat=quat,root=root,support=support,fps=20.)
                    r=score(m,policy,case,label,ref,extra=info);r['command_mps']=v
                    r['speed_success']=bool(r['complete'] and abs(r['speed_mps']-v)<=.05)
                    r['fidelity_pass']=info['lower_delta_p95_rad']<=.35 and info['upper_delta_max_rad']<1e-12
                    rows.append(r);save('contact_results.json',rows)
    save(args.stage+'_results.json',rows)
if __name__=='__main__':main()
