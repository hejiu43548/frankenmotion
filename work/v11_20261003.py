"""SONIC v1.1: exact 1751 layout and heading-normalized orientation."""
import os,sys,json,time,argparse
from pathlib import Path
BASE=Path('/home/pku/frankenmotion/outputs_amass/franken_eleven_20261003');OUT=BASE.parent/'franken_improve_20261003'
sys.path.insert(0,str(BASE/'code'));os.environ['ELEVEN_OUT']=str(BASE)
import transfer as tr
import numpy as np
import onnxruntime as ort
from scipy.spatial.transform import Rotation
class Policy(tr.rt.Policy):
    def __init__(self,mode=0):
        self.mode=mode;opts=ort.SessionOptions();opts.intra_op_num_threads=1;opts.inter_op_num_threads=1
        folder=tr.rt.ROOT/'gear_sonic_deploy/policy/sonic_v1_1'
        self.enc=ort.InferenceSession(str(folder/'model_encoder.onnx'),opts,providers=['CPUExecutionProvider']);self.dec=ort.InferenceSession(str(folder/'model_decoder.onnx'),opts,providers=['CPUExecutionProvider'])
        assert self.enc.get_inputs()[0].shape==[1,1751];assert self.dec.get_inputs()[0].shape==[1,994]
    def act(self,d,q,dq,quat,i):
        ids=np.minimum(i+np.arange(10)*(5 if self.mode==0 else 1),len(q)-1)
        rr=Rotation.from_quat(quat[ids][:,[1,2,3,0]]).as_matrix();robot=Rotation.from_quat(d.qpos[[4,5,6,3]]).as_matrix()
        heading=Rotation.from_euler('z',np.arctan2(robot[1,0],robot[0,0])).as_matrix();relative=heading.T@rr
        enc=np.zeros(1751,np.float32);enc[0]=self.mode
        if self.mode==0:
            enc[4:294]=q[ids][:,tr.rt.MJ_TO_IL].ravel();enc[294:584]=dq[ids][:,tr.rt.MJ_TO_IL].ravel();enc[584:644]=relative[:,:,:2].ravel()
        else:
            enc[911:1631]=self.joints[ids].ravel();enc[1631:1691]=relative[:,:,:2].ravel();enc[1691:1751]=q[ids][:,tr.rt.MJ_TO_IL][:,23:29].ravel()
        token=self.enc.run(None,{'obs_dict':enc[None]})[0].reshape(-1);obs=np.concatenate([token]+[np.asarray(h).ravel() for h in self.hist]).astype(np.float32)
        act=self.dec.run(None,{'obs_dict':obs[None]})[0].reshape(29);assert np.isfinite(act).all();act=np.clip(act,-20,20)
        return tr.rt.Q0+act[tr.rt.IL_TO_MJ]*tr.rt.SCALE,act
M=None;P=None
def init():
    global M,P
    M=tr.rt.load_model();P=[Policy(0),Policy(2)]
def run(row):
    folder=OUT/'v11_same_reference';folder.mkdir(parents=True,exist_ok=True);name=Path(row['path']).stem;dest=folder/(name+'.json')
    if dest.exists():return json.loads(dest.read_text())
    z=np.load(row['path']);ref=np.load(BASE/'simulation/task_adapter'/(name+'_reference.npz'));q,quat,root=ref['q'],ref['quat'],ref['root'];up=tr.upsample(q,quat,root);r=dict(row,modes={});start=time.monotonic()
    for policy in P:
        if policy.mode==2:policy.joints=tr.smpl_input(z,up[1])
        arrays,fall=tr.prior_run.rollout(M,policy,*up);ts=(np.arange(len(arrays['qpos']))+1)*.02;want=1+np.arange(len(q))*.05;metrics=None
        if fall is None and ts[-1]>=want[-1]-1e-7:
            ps=tr.get_positions(M,arrays['qpos']);p=np.stack([np.interp(want,ts,x) for x in ps.reshape(len(ps),-1).T],1).reshape(len(q),24,3);metrics=tr.measure(p,row['task'],tr.robot_height(M))
        r['modes'][str(policy.mode)]=dict(complete=metrics is not None,fall_time=fall,metrics=metrics)
        np.savez_compressed(folder/(name+f'_mode{policy.mode}.npz'),**arrays,time_s=ts)
    r['wall_s']=time.monotonic()-start;dest.write_text(json.dumps(r,default=lambda x:x.item()));return r
if __name__=='__main__':
    ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=6);ap.add_argument('--full',action='store_true');a=ap.parse_args()
    init();m=M;d=tr.mujoco.MjData(m);d.qpos[:]=m.qpos0
    for policy in P:
        if policy.mode==2:continue
        q=np.tile(tr.rt.Q0,(150,1));quat=np.tile([1.,0,0,0],(150,1));root=np.tile([0.,0,.8],(150,1))
        arrays,fall=tr.prior_run.rollout(m,policy,q,quat,root);assert fall is None,fall
        print('standing gate',policy.mode,arrays['qpos'][-1,:3],flush=True)
    rows=json.loads((BASE/'generated/task_adapter_manifest.json').read_text())
    if not a.full:rows=[r for r in rows if r['source'].endswith('_p0_s0')]
    from concurrent.futures import ProcessPoolExecutor
    results=[]
    with ProcessPoolExecutor(a.workers,initializer=init) as pool:
        for r in pool.map(run,rows):
            results.append(r)
            if len(results)%5==0:print('v1.1',len(results),'/',len(rows),flush=True)
    (OUT/('v11_full_results.json' if a.full else 'v11_probe_results.json')).write_text(json.dumps(results,indent=2,default=lambda x:x.item()))
