"""G1-specific retargeting and real-physics bounded reference optimization."""
import json,argparse
from pathlib import Path
import numpy as np
import mujoco
from scipy.optimize import least_squares
from scipy.ndimage import gaussian_filter1d
from scipy.spatial.transform import Rotation,Slerp
from g1_runtime import OUT,Q0,load_model,floor_align,Policy,simulate
SOURCES=Path('/home/pku/frankenmotion/outputs_amass/sim_posttrain_20261002/sources')
SCALES=np.array([.35,.20,.12,.20,.12,.06,np.pi/2])
NAMES=['leg_amplitude','arm_amplitude','hip_pitch_bias','knee_bias','ankle_pitch_bias','hip_roll_width','yaw_correction_rad']
def unit(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-9)

def retarget(m):
    out=OUT/'retarget';out.mkdir(parents=True,exist_ok=True)
    d=mujoco.MjData(m);limits=m.jnt_range[1:];segments=[]
    bid=lambda n:mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_BODY,n)
    for side,h,k,a,s,e,w in [('left',1,4,7,16,18,20),('right',2,5,8,17,19,21)]:
        segments += [(bid(side+'_hip_roll_link'),bid(side+'_knee_link'),h,k,1.5),
            (bid(side+'_knee_link'),bid(side+'_ankle_roll_link'),k,a,1.5),
            (bid(side+'_shoulder_roll_link'),bid(side+'_elbow_link'),s,e,1.),
            (bid(side+'_elbow_link'),bid(side+'_wrist_yaw_link'),e,w,1.)]
    feet=[bid(s+'_ankle_roll_link') for s in ['left','right']]
    assert all(min(a,b)>=0 for a,b,*_ in segments)
    for source in sorted(SOURCES.glob('*.npz')):
        target=out/source.name
        if target.exists():continue
        j=np.load(source)['joints_zup_m'].astype(float)
        left=unit(j[:,1]-j[:,2]);up=unit(j[:,12]-j[:,0]);fwd=unit(np.cross(left,up))
        yaw=np.unwrap(np.arctan2(fwd[:,1],fwd[:,0]));j=j@Rotation.from_euler('z',-yaw[0]).as_matrix().T;yaw-=yaw[0]
        quat=Rotation.from_euler('z',yaw[:,None]).as_quat()[:,[3,0,1,2]];qs=[];zs=[];err=[];prev=Q0.copy()
        for i,points in enumerate(j):
            dirs=[unit(points[b]-points[a]) for _,_,a,b,_ in segments]
            fd=[unit(points[b]-points[a]) for a,b in [(7,10),(8,11)]]
            d.qpos[:3]=[0,0,.85];d.qpos[3:7]=quat[i]
            def loss(q):
                d.qpos[7:]=q;mujoco.mj_forward(m,d)
                residual=[w*(unit(d.xpos[b]-d.xpos[a])-tar) for (a,b,_,_,w),tar in zip(segments,dirs)]
                for foot,direction in zip(feet,fd):
                    desired=unit(np.array([direction[0],direction[1],0.]))
                    residual.append(.4*(d.xmat[foot].reshape(3,3)[:,0]-desired))
                return np.concatenate(residual+[.08*(q-prev),.035*(q-Q0)])
            fit=least_squares(loss,np.clip(prev,limits[:,0]+1e-4,limits[:,1]-1e-4),bounds=(limits[:,0]+1e-5,limits[:,1]-1e-5),max_nfev=40,ftol=1e-5)
            err.append(np.mean(loss(fit.x)**2));floor_align(m,d);zs.append(d.qpos[2]);qs.append(fit.x);prev=fit.x
        root=j[:,0].copy();root[:,:2]-=root[0,:2];root[:,2]=gaussian_filter1d(zs,.65)
        np.savez_compressed(target,q=gaussian_filter1d(qs,.65,axis=0),root=root,quat=quat,fps=20.,ik_mse=err)
        print('retarget',source.stem,float(np.mean(err)),flush=True)

def reference(m,source,raw):
    z=np.load(source);q=z['q'].copy();root=z['root'];p=np.tanh(raw)*SCALES
    quat=z['quat'];yaw=np.unwrap(Rotation.from_quat(quat[:,[1,2,3,0]]).as_euler('xyz')[:,2])
    yaw+=np.linspace(0,p[6],len(q));quat=Rotation.from_euler('z',yaw[:,None]).as_quat()[:,[3,0,1,2]]
    q[:,:12]=Q0[:12]+(q[:,:12]-Q0[:12])*(1+p[0]);q[:,15:]=Q0[15:]+(q[:,15:]-Q0[15:])*(1+p[1])
    q[:,[0,6]]+=p[2];q[:,[3,9]]+=p[3];q[:,[4,10]]+=p[4];q[:,1]+=p[5];q[:,7]-=p[5]
    q=np.clip(q,m.jnt_range[1:,0]+1e-4,m.jnt_range[1:,1]-1e-4)
    a=np.linspace(0,1,21)[:-1];a=a*a*(3-2*a)
    q=np.concatenate([Q0+a[:,None]*(q[0]-Q0),q]);root=np.concatenate([np.repeat(root[:1],20,axis=0),root]);quat=np.concatenate([np.repeat(quat[:1],20,axis=0),quat])
    t=np.arange(len(q))/20;tt=np.arange(0,t[-1]-1e-8,.02)
    qi=np.stack([np.interp(tt,t,q[:,k]) for k in range(29)],axis=1)
    ri=np.stack([np.interp(tt,t,root[:,k]) for k in range(3)],axis=1)
    qu=Slerp(t,Rotation.from_quat(quat[:,[1,2,3,0]]))(tt).as_quat()[:,[3,0,1,2]]
    d=mujoco.MjData(m)
    for i in range(len(qi)):
        d.qpos[:3]=ri[i];d.qpos[3:7]=qu[i];d.qpos[7:]=qi[i];floor_align(m,d);ri[i,2]=d.qpos[2]
    return qi,qu,ri

def evaluate(m,policy,source,raw,label,save=False,seed=None):
    q,quat,root=reference(m,source,raw);s,c,fall=simulate(m,policy,q,quat,root,seed)
    yaw=np.unwrap(Rotation.from_quat(s[:,[4,5,6,3]]).as_euler('xyz')[:,2]);heading=(yaw[50:-1]+yaw[51:])/2
    delta=np.diff(s[50:,:2],axis=0);speed=float(np.mean(delta[:,0]*np.cos(heading)+delta[:,1]*np.sin(heading))/.02)
    angle=float(np.rad2deg(yaw[-1]-yaw[50]));target_speed=.35 if source.stem.startswith('01') else (.55 if source.stem.startswith('03') else .65)
    target_angle=60. if source.stem.startswith('03') else 0.;rmse=float(np.sqrt(np.mean((s[:,7:]-q)**2)))
    duration=len(s)*.02;survival=(fall or duration)/duration
    score=6*survival-3*abs(speed-target_speed)-.012*abs(angle-target_angle)-rmse-.25*np.mean(np.tanh(raw)**2)-(3 if fall else 0)
    reference_yaw=np.unwrap(Rotation.from_quat(quat[:,[1,2,3,0]]).as_euler('xyz')[:,2])
    r=dict(label=label,robot='Unitree G1 29 DOF',score=float(score),fall=fall is not None,fall_time_s=fall,
        forward_speed_mps=speed,target_speed_mps=target_speed,turn_deg=angle,target_turn_deg=target_angle,
        reference_turn_deg=float(np.rad2deg(reference_yaw[-1]-reference_yaw[50])),joint_rmse_rad=rmse,
        simulation_seconds=duration,raw_parameters=np.asarray(raw).tolist(),bounded_parameters=(np.tanh(raw)*SCALES).tolist(),perturb_seed=seed)
    if save:
        np.savez_compressed(OUT/'rollouts'/f'{label}.npz',qpos=s,reference_qpos=np.concatenate([root,quat,q],axis=1),fps=50.,contacts=c)
        (OUT/'rollouts'/f'{label}.json').write_text(json.dumps(r,indent=2))
    with (OUT/'search.jsonl').open('a') as f:f.write(json.dumps(r)+'\n')
    print(json.dumps(r),flush=True);return r

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--validate',action='store_true');ap.add_argument('--rounds',type=int,default=2);ap.add_argument('--population',type=int,default=8);args=ap.parse_args()
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'rollouts').mkdir(exist_ok=True)
    m=load_model();policy=Policy();retarget(m)
    if args.validate:
        results=[]
        for row in json.loads((OUT/'results.json').read_text()):
            for seed in [1301,2702]:
                for variant in ['before','after']:
                    results.append(evaluate(m,policy,OUT/'retarget'/f"{row['case']}.npz",row[variant]['raw_parameters'],f"{row['case']}_{variant}_heldout_{seed}",seed=seed))
        (OUT/'perturbation_validation.json').write_text(json.dumps(results,indent=2));return
    if (OUT/'results.json').exists():raise FileExistsError('Do not overwrite completed G1 run')
    rng=np.random.default_rng(20261003);results=[]
    for source in sorted((OUT/'retarget').glob('*.npz')):
        name=source.stem;base=evaluate(m,policy,source,np.zeros(7),name+'_before',True);best=base;best_raw=np.zeros(7)
        mean=np.zeros(7);mean[6]=np.arctanh(np.clip((base['target_turn_deg']-base['reference_turn_deg'])/90,-.9,.9))
        r=evaluate(m,policy,source,mean,name+'_command_init')
        if r['score']>best['score']:best=r;best_raw=mean.copy()
        for gen in range(args.rounds):
            noise=rng.normal(size=(args.population,7))*.55*(.7**gen);noise[:,6]*=.18
            samples=mean+noise;reports=[evaluate(m,policy,source,raw,f'{name}_g{gen}_k{k}') for k,raw in enumerate(samples)]
            for raw,r in zip(samples,reports):
                if r['score']>best['score']:best=r;best_raw=raw.copy()
            rewards=np.array([r['score'] for r in reports]);w=np.exp((rewards-rewards.max())/.4);w/=w.sum()
            mean=.25*mean+.75*np.sum(w[:,None]*samples,axis=0)
        after=evaluate(m,policy,source,best_raw,name+'_after',True)
        results.append(dict(case=name,before=base,after=after))
        (OUT/'results.json').write_text(json.dumps(results,indent=2))
        np.savez_compressed(OUT/f'{name}_editor.npz',selected_raw=best_raw,search_mean=mean,scales=SCALES,parameter_names=NAMES)
    (OUT/'status.json').write_text(json.dumps(dict(status='completed',robot='Unitree G1',method='per-clip reward-weighted black-box reference search',generator_frozen=True,tracker_frozen=True)))

if __name__=='__main__':main()
