"""Source-derived Cartesian IK ablation. Baseline confirmation is never modified."""
import os,sys,json,time,argparse
from pathlib import Path
BASE=Path('/home/pku/frankenmotion/outputs_amass/franken_eleven_20261003')
OUT=Path('/home/pku/frankenmotion/outputs_amass/franken_improve_20261003')
sys.path.insert(0,str(BASE/'code'));os.environ['ELEVEN_OUT']=str(BASE)
import transfer as tr
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
import mujoco

def convert(m,z,weight):
    human=z['joints_zup_m'].astype(float);poses=z['poses_axisangle'].astype(float)
    side=human[0,1]-human[0,2];yaw0=np.arctan2(side[1],side[0])-np.pi/2;rz=Rotation.from_euler('z',-yaw0)
    human=rz.apply(human.reshape(-1,3)).reshape(human.shape)
    R=rz*Rotation.from_rotvec(poses[:,:3])*Rotation.from_quat([.5,.5,.5,.5]).inv();quat=R.as_quat()[:,[3,0,1,2]]
    scale=tr.robot_height(m)/float(z['human_height']);root=human[:,0].copy();root[:,:2]=(root[:,:2]-root[:1,:2])*scale;root[:,2]=(root[:,2]-root[0,2])*scale+.8
    segs=[(1,4,2.),(4,7,2.),(2,5,2.),(5,8,2.),(16,18,1.5),(18,20,1.5),(17,19,1.5),(19,21,1.5)]
    d=mujoco.MjData(m);prev=tr.rt.Q0.copy();qs=[];res=[];lo=m.jnt_range[1:,0]+1e-4;hi=m.jnt_range[1:,1]-1e-4
    for t,h in enumerate(human):
        d.qpos[:7]=np.r_[root[t],quat[t]];targets=[tr.unit(h[b]-h[a]) for a,b,w in segs];torso=tr.unit((h[16]+h[17])/2-h[0]);memo={}
        def calc(q):
            if 'q' in memo and np.array_equal(q,memo['q']):return memo['r'],memo['j']
            d.qpos[7:]=q;mujoco.mj_forward(m,d);p,j=tr.markers(m,d,True);rs=[];js=[]
            for (a,b,w),tar in zip(segs,targets):
                v=p[b]-p[a];length=np.linalg.norm(v);u=v/max(length,1e-8)
                rs.append(w*(u-tar));js.append(w*(np.eye(3)-np.outer(u,u))@(j[b]-j[a])/max(length,1e-8))
            v=(p[16]+p[17])/2-p[0];length=np.linalg.norm(v);u=v/length
            rs.append(2*(u-torso));js.append(2*(np.eye(3)-np.outer(u,u))@((j[16]+j[17])/2)/length)
            for idx in [20,21]:
                rs.append(weight*((p[idx]-p[0])-(h[idx]-h[0])*scale));js.append(weight*j[idx])
            rs.extend([.06*(q-prev),.02*(q-tr.rt.Q0)]);js.extend([.06*np.eye(29),.02*np.eye(29)])
            r=np.concatenate(rs);J=np.concatenate(js);memo.update(q=q.copy(),r=r,j=J);return r,J
        fit=least_squares(lambda q:calc(q)[0],np.clip(prev,lo,hi),jac=lambda q:calc(q)[1],bounds=(lo,hi),max_nfev=30,ftol=1e-4,xtol=1e-4)
        qs.append(fit.x);res.append(float(np.linalg.norm(fit.fun)));prev=fit.x
    q=np.array(qs);d.qpos[:]=np.r_[root[0],quat[0],q[0]];tr.rt.floor_align(m,d);root[:,2]+=d.qpos[2]-root[0,2]
    return q,quat,root

def init():tr.init()
def run(pair):
    row,w=pair;folder=OUT/'ik_sweep';folder.mkdir(exist_ok=True,parents=True);name=f"{Path(row['path']).stem}_w{w:g}";dest=folder/(name+'.json')
    if dest.exists():return json.loads(dest.read_text())
    m=tr.MODEL;z=np.load(row['path']);start=time.monotonic();q,quat,root=convert(m,z,w);h=tr.robot_height(m)
    states=np.c_[root,quat,q];reference=tr.measure(tr.get_positions(m,states),row['task'],h)
    arrays,fall=tr.prior_run.rollout(m,tr.POLICIES[0],*tr.upsample(q,quat,root));ts=(np.arange(len(arrays['qpos']))+1)*.02;want=1+np.arange(len(q))*.05
    actual=None
    if fall is None and ts[-1]>=want[-1]-1e-7:
        p=tr.get_positions(m,arrays['qpos']);pp=np.stack([np.interp(want,ts,x) for x in p.reshape(len(p),-1).T],1).reshape(len(q),24,3);actual=tr.measure(pp,row['task'],h)
    r=dict(row,weight=w,g1=reference,actual=actual,fall=fall,wall_s=time.monotonic()-start)
    np.savez_compressed(folder/(name+'.npz'),reference_qpos=states,**arrays,time_s=ts)
    dest.write_text(json.dumps(r));return r
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=6);args=ap.parse_args()
    rows=json.loads((BASE/'generated/task_adapter_manifest.json').read_text());rows=[r for r in rows if r['source'].endswith('_p0_s0') and r['task'] in ['raise_hand','reach','strike','wave','lean']]
    protocol=dict(source='Post-hoc development subset of prior 880; NOT fresh validation',variants=[4,10,25],tasks=sorted(set(r['task'] for r in rows)),selection='Compare aggregate dev kinematic error and physical execution, then freeze and test fresh seeds',goal='Match source wrist position relative to pelvis with morphology scale; no numeric target injected into IK')
    OUT.mkdir(parents=True,exist_ok=True);(OUT/'ik_protocol.json').write_text(json.dumps(protocol,indent=2))
    from concurrent.futures import ProcessPoolExecutor
    results=[]
    with ProcessPoolExecutor(args.workers,initializer=init) as pool:
        for r in pool.map(run,[(r,w) for w in [4,10,25] for r in rows]):
            results.append(r);print(len(results),r['task'],r['weight'],r['g1']['quantity'],None if r['actual'] is None else r['actual']['quantity'],flush=True)
    (OUT/'ik_sweep_results.json').write_text(json.dumps(results,indent=2))
