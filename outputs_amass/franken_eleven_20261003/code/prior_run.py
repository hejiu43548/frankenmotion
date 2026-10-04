"""Isolated G1 experiments. Frozen FrankenMotion sources and SONIC; no hardware."""
import os,time,json,hashlib,argparse
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation,Slerp
import g1_runtime as rt
import original_experiment as ex

OUT=Path(os.environ['DIAG_OUT'])
SOURCES=Path('/home/pku/frankenmotion/outputs_amass/g1_sim_20261002/retarget')
CASES=['01_walk_speed_v1','04_walk_wave_v2']

def write(name,data):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True)
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(data,indent=2));temp.replace(p)

def reference(m,case,cadence=1.,amplitude=1.):
    q,quat,root=ex.reference(m,SOURCES/(case+'.npz'),np.zeros(7))
    # Keep the 1 s entry identical. Retime only the original action; duration is explicit.
    original=np.arange(len(q))*.02
    task_duration=original[-1]-1.
    task_t=np.arange(0,task_duration/cadence+1e-9,.02)
    sample=np.concatenate([np.arange(50)*.02,1+task_t*cadence])
    qi=np.stack([np.interp(sample,original,q[:,j]) for j in range(29)],axis=1)
    ri=np.stack([np.interp(sample,original,root[:,j]) for j in range(3)],axis=1)
    qu=Slerp(original,Rotation.from_quat(quat[:,[1,2,3,0]]))(sample).as_quat()[:,[3,0,1,2]]
    # Scale locomotion joints only, smoothly introduced over fixed entry.
    ramp=np.ones(len(qi));ramp[:50]=np.linspace(0,1,50)**2*(3-2*np.linspace(0,1,50))
    factor=1+(amplitude-1)*ramp
    qi[:,:12]=rt.Q0[:12]+(qi[:,:12]-rt.Q0[:12])*factor[:,None]
    qi=np.clip(qi,m.jnt_range[1:,0]+1e-4,m.jnt_range[1:,1]-1e-4)
    return qi,qu,ri

def rollout(m,policy,q,quat,root,seed=None):
    d=mujoco.MjData(m);d.qpos[:3]=root[0];d.qpos[3:7]=quat[0];d.qpos[7:]=q[0]
    rt.floor_align(m,d)
    if seed is not None:
        rng=np.random.default_rng(seed);d.qvel[:2]=rng.uniform(-.03,.03,2);d.qvel[3:6]=rng.uniform(-.03,.03,3)
    policy.reset(d);dq=np.gradient(q,.02,axis=0)
    states=[];vel=[];contacts=[];slips=[];actions=[];fall=None
    feet=[m.body(s+'_ankle_roll_link').id for s in ['left','right']]
    for i in range(len(q)):
        target,act=policy.act(d,q,dq,quat,i)
        for _ in range(10):
            d.ctrl[:]=np.clip(rt.KP*(target-d.qpos[7:])-rt.KD*d.qvel[6:],-rt.EFF,rt.EFF)
            mujoco.mj_step(m,d)
        mujoco.mj_forward(m,d);policy.update(d,act)
        states.append(d.qpos.copy());vel.append(d.qvel.copy());actions.append(act)
        touching=set()
        for ct in d.contact:
            b1,b2=int(m.geom_bodyid[ct.geom1]),int(m.geom_bodyid[ct.geom2])
            if b1==0 and b2 in feet:touching.add(b2)
            if b2==0 and b1 in feet:touching.add(b1)
        foot_slip=[]
        for foot in touching:
            v=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,foot,v,0)
            foot_slip.append(float(np.linalg.norm(v[3:5])))
        contacts.append(len(touching));slips.append(float(np.mean(foot_slip)) if foot_slip else np.nan)
        tilt=np.linalg.norm(Rotation.from_quat(d.qpos[[4,5,6,3]]).as_euler('xyz')[:2])
        if d.qpos[2]<.35 or tilt>np.pi/3 or not np.isfinite(d.qpos).all():
            fall=(i+1)*.02;break
    return dict(qpos=np.array(states),qvel=np.array(vel),action=np.array(actions),contacts=np.array(contacts),contact_body_speed=np.array(slips)),fall

def evaluate(m,policy,case,cadence,amplitude,label,seed=None,root_only_scale=None):
    start=time.monotonic();q,quat,root=reference(m,case,cadence,amplitude)
    if root_only_scale is not None:
        root[50:,:2]=root[50,:2]+root_only_scale*(root[50:,:2]-root[50,:2])
    arrays,fall=rollout(m,policy,q,quat,root,seed)
    s=arrays['qpos'];complete=fall is None and len(s)==len(q)
    actual=None;heading_speed=None;turn=None;drift=None
    if complete:
        # Both fixed-start forward speed and heading-following speed are retained.
        yaw=np.unwrap(Rotation.from_quat(s[:,[4,5,6,3]]).as_euler('xyz')[:,2])
        axis=Rotation.from_quat(quat[50,[1,2,3,0]]).apply([1.,0,0])[:2]
        actual=float((s[-1,:2]-s[50,:2])@axis/((len(s)-51)*.02))
        delta=np.diff(s[50:,:2],axis=0);head=(yaw[50:-1]+yaw[51:])/2
        heading_speed=float(np.mean(delta[:,0]*np.cos(head)+delta[:,1]*np.sin(head))/.02)
        turn=float(np.rad2deg(yaw[-1]-yaw[50]));drift=float(np.max(np.abs(yaw[50:]-yaw[50]))*180/np.pi)
    r=dict(label=label,case=case,cadence=cadence,amplitude=amplitude,seed=seed,
           root_only_scale=root_only_scale,complete=complete,fall_time_s=fall,
           speed_mps=actual,heading_speed_mps=heading_speed,turn_deg=turn,max_heading_drift_deg=drift,
           joint_rmse_rad=float(np.sqrt(np.mean((s[:,7:]-q[:len(s)])**2))),
           joint_speed_max=float(np.max(np.abs(np.gradient(q,.02,axis=0)))),
           contact_body_speed_mean=float(np.nanmean(arrays['contact_body_speed'][50:])),
           duration_s=(len(q)-50)*.02,wall_s=time.monotonic()-start)
    np.savez_compressed(OUT/'rollouts'/(label+'.npz'),**arrays,reference_qpos=np.c_[root,quat,q],fps=50.)
    write('rollouts/'+label+'.json',r)
    with (OUT/'ledger.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
    print(json.dumps(r),flush=True)
    return r

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--stage',choices=['probe','sweep','validate'],required=True);args=ap.parse_args()
    OUT.mkdir(exist_ok=True,parents=True);(OUT/'rollouts').mkdir(exist_ok=True)
    m=rt.load_model();policy=rt.Policy()
    if args.stage=='probe':
        for factor in [1.,2.,0.]:evaluate(m,policy,CASES[0],1.,1.,'root_probe_'+str(factor),root_only_scale=factor)
        base=np.load(OUT/'rollouts/root_probe_1.0.npz')
        comparison={str(f):{k:float(np.max(np.abs(base[k]-np.load(OUT/('rollouts/root_probe_'+str(f)+'.npz'))[k]))) for k in ['qpos','action']} for f in [2.,0.]}
        write('root_invariance.json',comparison)
    elif args.stage=='sweep':
        grid=[(c,a) for c in [.6,.8,1.,1.2,1.4,1.6,1.8] for a in [.8,1.,1.2,1.4]]
        write('protocol.json',dict(cases=CASES,grid=grid,commands=[.2,.3,.4,.5,.6],tolerance_mps=.05,entry_s=1.,note='Per-source calibration, not generator training. Duration changes with cadence; all failures retained. Actual fixed-heading metric. Contact-body speed is a proxy, not contact-point slip.'))
        rows=[]
        for case in CASES:
            for c,a in grid:
                label=f'{case}_c{c:.1f}_a{a:.1f}'
                p=OUT/'rollouts'/(label+'.json')
                r=json.loads(p.read_text()) if p.exists() else evaluate(m,policy,case,c,a,label)
                rows.append(r);write('sweep.json',rows)
        write('status.json',dict(stage='sweep_complete',count=len(rows)))
    else:
        rows=json.loads((OUT/'sweep.json').read_text());selected=[]
        for case in CASES:
            candidates=[r for r in rows if r['case']==case and r['complete'] and r['max_heading_drift_deg']<15 and r['joint_speed_max']<35]
            for cmd in [.2,.3,.4,.5,.6]:
                for method in ['cadence','cadence_amplitude']:
                    pool=[r for r in candidates if method!='cadence' or abs(r['amplitude']-1)<1e-6]
                    if not pool:continue
                    best=min(pool,key=lambda r:abs(r['speed_mps']-cmd))
                    for seed in [1301,2702]:
                        label=f'{case}_{method}_v{cmd:.1f}_s{seed}'
                        r=evaluate(m,policy,case,best['cadence'],best['amplitude'],label,seed)
                        r.update(command_mps=cmd,method=method,calibration_label=best['label'],success=bool(r['complete'] and abs(r['speed_mps']-cmd)<=.05 and r['max_heading_drift_deg']<15))
                        selected.append(r);write('validation.json',selected)
        write('status.json',dict(stage='validation_complete',count=len(selected)))

if __name__=='__main__':main()
