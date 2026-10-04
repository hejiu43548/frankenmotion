"""Native-pose direction IK, measured G1 references, two frozen SONIC execution paths."""
import os,json,argparse,time
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','1');os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
OUT=Path(os.environ.get('ELEVEN_OUT','/home/pku/frankenmotion/outputs_amass/franken_eleven_20261003'))
os.environ['DIAG_OUT']=str(OUT)
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation,Slerp
from scipy.optimize import least_squares
import g1_runtime as rt
import prior_run
from partner_metrics import measure as partner_measure
HH=1.2701193988323212
NAMES=['Pelvis','L_Hip','R_Hip','Torso1','L_Knee','R_Knee','Torso2','L_Ankle','R_Ankle','Torso','L_Toe','R_Toe','Neck','L_Collar','R_Collar','Head','L_Shoulder','R_Shoulder','L_Elbow','R_Elbow','L_Wrist','R_Wrist','L_Hand','R_Hand']
JOINTS={1:'left_hip_pitch_joint',2:'right_hip_pitch_joint',4:'left_knee_joint',5:'right_knee_joint',7:'left_ankle_roll_joint',8:'right_ankle_roll_joint',16:'left_shoulder_pitch_joint',17:'right_shoulder_pitch_joint',18:'left_elbow_joint',19:'right_elbow_joint',20:'left_wrist_yaw_joint',21:'right_wrist_yaw_joint'}
MODEL=None;POLICIES=None
def init():
    global MODEL,POLICIES
    MODEL=rt.load_model();POLICIES=(rt.Policy(),Mode2())
def markers(m,d,jac=False):
    p=np.zeros((24,3));p[0]=d.xpos[m.body('pelvis').id];js=np.zeros((24,3,29))
    for idx,name in JOINTS.items():
        j=m.joint(name);p[idx]=d.xanchor[j.id]
        if jac:
            jp=np.zeros((3,m.nv));mujoco.mj_jac(m,d,jp,None,p[idx],j.bodyid[0]);js[idx]=jp[:,6:]
    return (p,js) if jac else p
def robot_height(m):
    d=mujoco.MjData(m);d.qpos[:]=m.qpos0;mujoco.mj_forward(m,d);p=markers(m,d)
    return float((p[16,2]+p[17,2]-p[7,2]-p[8,2])/2)
def unit(x):return x/np.maximum(np.linalg.norm(x,axis=-1,keepdims=True),1e-8)
def prepare_reference(m,z):
    human=z['joints_zup_m'].astype(float);poses=z['poses_axisangle'].astype(float)
    side=human[0,1]-human[0,2];yaw0=np.arctan2(side[1],side[0])-np.pi/2;rz=Rotation.from_euler('z',-yaw0)
    human=rz.apply(human.reshape(-1,3)).reshape(human.shape)
    R=rz*Rotation.from_rotvec(poses[:,:3])*Rotation.from_quat([.5,.5,.5,.5]).inv();quat=R.as_quat()[:,[3,0,1,2]]
    scale=robot_height(m)/float(z['human_height']);root=human[:,0].copy();root[:,:2]=(root[:,:2]-root[:1,:2])*scale;root[:,2]=(root[:,2]-root[0,2])*scale+.8
    segs=[(1,4,2.),(4,7,2.),(2,5,2.),(5,8,2.),(16,18,1.5),(18,20,1.5),(17,19,1.5),(19,21,1.5)]
    d=mujoco.MjData(m);prev=rt.Q0.copy();qs=[];errors=[]
    lo=m.jnt_range[1:,0]+1e-4;hi=m.jnt_range[1:,1]-1e-4
    for t,h in enumerate(human):
        d.qpos[:7]=np.r_[root[t],quat[t]];targets=[unit(h[b]-h[a]) for a,b,w in segs];torso=unit((h[16]+h[17])/2-h[0]);memo={}
        def calc(q):
            if 'q' in memo and np.array_equal(q,memo['q']):return memo['r'],memo['j']
            d.qpos[7:]=q;mujoco.mj_forward(m,d);p,j=markers(m,d,True);rs=[];jac=[]
            for (a,b,w),tar in zip(segs,targets):
                v=p[b]-p[a];length=np.linalg.norm(v);u=v/max(length,1e-8)
                rs.append(w*(u-tar));jac.append(w*(np.eye(3)-np.outer(u,u))@(j[b]-j[a])/max(length,1e-8))
            v=(p[16]+p[17])/2-p[0];length=np.linalg.norm(v);u=v/length
            rs.append(2*(u-torso));jac.append(2*(np.eye(3)-np.outer(u,u))@((j[16]+j[17])/2)/length)
            rs.extend([.06*(q-prev),.02*(q-rt.Q0)]);jac.extend([.06*np.eye(29),.02*np.eye(29)])
            r=np.concatenate(rs);J=np.concatenate(jac);memo.update(q=q.copy(),r=r,j=J);return r,J
        fit=least_squares(lambda q:calc(q)[0],np.clip(prev,lo,hi),jac=lambda q:calc(q)[1],bounds=(lo,hi),max_nfev=20,ftol=1e-4,xtol=1e-4)
        qs.append(fit.x);errors.append(float(np.sqrt(np.mean(calc(fit.x)[0][:27]**2))));prev=fit.x
    q=np.array(qs);d.qpos[:]=np.r_[root[0],quat[0],q[0]];rt.floor_align(m,d);root[:,2]+=d.qpos[2]-root[0,2]
    return q,quat,root,dict(direction_residual_mean=float(np.mean(errors)),robot_height_m=robot_height(m),human_height_m=float(z['human_height']))
def upsample(q,quat,root):
    n=len(q);a=np.linspace(0,1,21)[:-1];a=a*a*(3-2*a)
    q=np.r_[rt.Q0+a[:,None]*(q[0]-rt.Q0),q];root=np.r_[np.repeat(root[:1],20,axis=0),root];quat=np.r_[np.repeat(quat[:1],20,axis=0),quat]
    t=np.arange(len(q))*.05;tt=np.arange(0,t[-1]-1e-8,.02)
    qi=np.stack([np.interp(tt,t,x) for x in q.T],1);ri=np.stack([np.interp(tt,t,x) for x in root.T],1)
    qu=Slerp(t,Rotation.from_quat(quat[:,[1,2,3,0]]))(tt).as_quat()[:,[3,0,1,2]]
    return qi,qu,ri
def smpl_input(z,quat50):
    poses=z['poses_axisangle'];sk=np.load(OUT/'official_human_skeleton.npz');J=sk['J'].reshape(-1,3);parents=sk['parents'].reshape(-1)
    rr=Rotation.from_rotvec(np.pad(poses,((0,0),(0,99))).reshape(-1,3)).as_matrix().reshape(-1,55,3,3);g=[];p=[]
    for i in range(55):
        if i==0:g.append(rr[:,i]);p.append(np.repeat(J[:1],len(poses),axis=0))
        else:
            par=int(parents[i]);g.append(g[par]@rr[:,i]);p.append(p[par]+np.einsum('nij,j->ni',g[par],J[i]-J[par]))
    joints=np.stack(p,1)[:,list(range(22))+[39,54]];r=Rotation.from_rotvec(poses[:,:3])*Rotation.from_quat([.5,.5,.5,.5]).inv()
    local=np.einsum('nij,nkj->nki',r.as_matrix().transpose(0,2,1),joints);local=np.r_[np.repeat(local[:1],20,0),local]
    t=np.arange(len(local))*.05;tt=np.arange(len(quat50))*.02
    return np.stack([np.interp(tt,t,x) for x in local.reshape(len(local),-1).T],1).reshape(-1,24,3)
class Mode2(rt.Policy):
    def act(self,d,q,dq,quat,i):
        ids=np.minimum(i+np.arange(10),len(q)-1);rr=Rotation.from_quat(quat[ids][:,[1,2,3,0]]).as_matrix();robot=Rotation.from_quat(d.qpos[[4,5,6,3]]).as_matrix()
        enc=np.zeros(1762,np.float32);enc[0]=2;enc[922:1642]=self.joints[ids].ravel();enc[1642:1702]=(robot.T@rr)[:,:,:2].ravel();enc[1702:]=q[ids][:,rt.MJ_TO_IL][:,23:29].ravel()
        token=self.enc.run(None,{'obs_dict':enc[None]})[0].reshape(-1);obs=np.concatenate([token]+[np.asarray(h).ravel() for h in self.hist]).astype(np.float32)
        act=self.dec.run(None,{'obs_dict':obs[None]})[0].reshape(29)
        if not np.isfinite(act).all():raise FloatingPointError('Nonfinite policy')
        act=np.clip(act,-20,20);return rt.Q0+act[rt.IL_TO_MJ]*rt.SCALE,act
def measure(p,task,height):
    scale=HH/height
    if task in ['raise_hand','back_walk','lean']:
        pelvis=p[:,0];w=p[:,21]
        if task=='raise_hand':
            s=(w-pelvis)[:,2];q=np.quantile(s,.95);event=np.max(s[1:]-np.minimum.accumulate(s[:-1]))>=.08/scale
        elif task=='back_walk':
            q=-(pelvis[-1,0]-pelvis[0,0])/((len(p)-1)/20);side=p[:,1,:2]-p[:,2,:2];yaw=np.unwrap(np.arctan2(side[:,1],side[:,0]));stride=np.ptp((p[:,8]-pelvis)[:,0]);angle=np.arctan2(np.sin(yaw[-1]-yaw[0]),np.cos(yaw[-1]-yaw[0]));event=q*((len(p)-1)/20)>=1.2/scale and abs(angle)<=.65 and stride>=.2/scale
        else:
            v=(p[:,16]+p[:,17])/2-pelvis;pitch=np.arctan2(v[:,0],v[:,2]);q=np.quantile(pitch[20:59],.9);event=q-pitch[0]>=.25 and pitch[-1]-pitch[0]>=.2 and pelvis[0,2]-pelvis[:,2].min()<=.2/scale and np.max(np.linalg.norm(pelvis[:,:2]-pelvis[0,:2],axis=-1))<=.3/scale
        out=dict(quantity=float(q)*(1 if task=='lean' else scale),event_pass=bool(event))
    else:
        out=partner_measure(task,p,NAMES,distance_ratio=1/scale);out['quantity']*=1 if task=='turn' else scale
    if task=='walk':out['net_forward_speed']=(p[-1,0,0]-p[0,0,0])/((len(p)-1)/20)*scale
    return out
def get_positions(m,states):
    d=mujoco.MjData(m);ps=[]
    for s in states:d.qpos[:]=s;mujoco.mj_forward(m,d);ps.append(markers(m,d))
    return np.array(ps)
def work(item):
    row,tag=item;m=MODEL;folder=OUT/'simulation'/tag;label=Path(row['path']).stem;result=folder/(label+'.json')
    if result.exists():return json.loads(result.read_text())
    start=time.monotonic();z=np.load(row['path']);r=dict(row);r['modes']={}
    try:
        refpath=folder/(label+'_reference.npz')
        if refpath.exists():
            refz=np.load(refpath);q,quat,root=refz['q'],refz['quat'],refz['root'];info=json.loads(str(refz['info']))
        else:
            q,quat,root,info=prepare_reference(m,z);np.savez_compressed(refpath,q=q,quat=quat,root=root,info=json.dumps(info))
        r['conversion']=info;ref=upsample(q,quat,root);r['g1_reference']=measure(get_positions(m,np.c_[root,quat,q]),row['task'],info['robot_height_m'])
        for mode,policy in enumerate(POLICIES):
            if mode==1:policy.joints=smpl_input(z,ref[1])
            arrays,fall=prior_run.rollout(m,policy,*ref,None);p=get_positions(m,arrays['qpos']);time_s=(np.arange(len(p))+1)*.02;desired=1+np.arange(len(q))*.05
            complete=fall is None and time_s[-1]>=desired[-1]-1e-7;metrics=None
            if complete:
                flat=p.reshape(len(p),-1);sampled=np.stack([np.interp(desired,time_s,x) for x in flat.T],1).reshape(len(q),24,3);metrics=measure(sampled,row['task'],info['robot_height_m'])
            r['modes'][str(mode*2)]=dict(complete=complete,fall_time=fall,metrics=metrics)
            np.savez_compressed(folder/(label+f'_mode{mode*2}.npz'),**arrays,time_s=time_s)
    except Exception as exc:r['error']=repr(exc)
    r['wall_s']=time.monotonic()-start;temp=result.with_suffix('.tmp');temp.write_text(json.dumps(r,indent=2,default=lambda x:x.item()));temp.replace(result);return r
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--tag',default='task_adapter');ap.add_argument('--workers',type=int,default=6);args=ap.parse_args()
    rows=json.loads((OUT/'generated'/(args.tag+'_manifest.json')).read_text());(OUT/'simulation'/args.tag).mkdir(parents=True,exist_ok=True)
    from concurrent.futures import ProcessPoolExecutor
    results=[]
    with ProcessPoolExecutor(args.workers,initializer=init) as pool:
        for r in pool.map(work,[(r,args.tag) for r in rows]):
            results.append(r)
            if len(results)%10==0:print('completed',len(results),'/',len(rows),'errors',sum('error' in x for x in results),flush=True)
    (OUT/'simulation'/(args.tag+'_results.json')).write_text(json.dumps(results,indent=2,default=lambda x:x.item()))
if __name__=='__main__':main()
