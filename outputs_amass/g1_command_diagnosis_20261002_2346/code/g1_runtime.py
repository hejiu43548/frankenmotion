"""CPU ONNX bridge following NVIDIA's released G1 observation/PD contract."""
import os
os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
from collections import deque
import numpy as np
import mujoco
import onnxruntime as ort
from scipy.spatial.transform import Rotation
ROOT=Path('/home/pku/frankenmotion/work/g1_sonic_official')
OUT=Path('/home/pku/frankenmotion/outputs_amass/g1_sim_20261002')
XML=ROOT/'gear_sonic/data/assets/robot_description/mjcf/g1_29dof_rev_1_0.xml'
POLICY=ROOT/'gear_sonic_deploy/policy/release'
MJ_TO_IL=np.array([0,6,12,1,7,13,2,8,14,3,9,15,22,4,10,16,23,5,11,17,24,18,25,19,26,20,27,21,28])
IL_TO_MJ=np.argsort(MJ_TO_IL)
Q0=np.array([-.312,0,0,.669,-.363,0]*2+[0,0,0]+[.2,.2,0,.6,0,0,0]+[.2,-.2,0,.6,0,0,0])
ARM=np.array([.025101925,.025101925,.010177520,.025101925,.003609725,.003609725]*2+[.010177520,.003609725,.003609725]+[.003609725]*5+[.00425]*2+[.003609725]*5+[.00425]*2)
EFF=np.array([139,139,88,139,25,25]*2+[88,25,25]+[25]*5+[5]*2+[25]*5+[5]*2)
KP=ARM*(20*np.pi)**2; KD=4*ARM*(20*np.pi);SCALE=.25*EFF/KP
KP[[4,5,10,11,13,14]]*=2;KD[[4,5,10,11,13,14]]*=2

def load_model():
    m=mujoco.MjModel.from_xml_path(str(XML));m.opt.timestep=.002
    assert m.nq==36 and m.nu==29
    # Training actuator armatures, same motor constants as official PD contract.
    m.dof_armature[6:]=ARM
    return m

def floor_align(m,d):
    mujoco.mj_forward(m,d);low=[]
    for g in range(m.ngeom):
        name=mujoco.mj_id2name(m,mujoco.mjtObj.mjOBJ_BODY,int(m.geom_bodyid[g])) or ''
        if 'ankle_roll' not in name or not (m.geom_contype[g] or m.geom_conaffinity[g]):continue
        typ=m.geom_type[g];p=d.geom_xpos[g];R=d.geom_xmat[g].reshape(3,3);s=m.geom_size[g]
        if typ==mujoco.mjtGeom.mjGEOM_SPHERE:low.append(p[2]-s[0])
        elif typ==mujoco.mjtGeom.mjGEOM_BOX:low.append(p[2]-np.abs(R[2]).dot(s))
        elif typ==mujoco.mjtGeom.mjGEOM_MESH:
            k=m.geom_dataid[g];v=m.mesh_vert[m.mesh_vertadr[k]:m.mesh_vertadr[k]+m.mesh_vertnum[k]]
            low.append((v@R.T+p)[:,2].min())
    assert low
    d.qpos[2]-=min(low);mujoco.mj_forward(m,d)

class Policy:
    def __init__(self):
        opts=ort.SessionOptions();opts.intra_op_num_threads=2;opts.inter_op_num_threads=1
        self.enc=ort.InferenceSession(str(POLICY/'model_encoder.onnx'),opts,providers=['CPUExecutionProvider'])
        self.dec=ort.InferenceSession(str(POLICY/'model_decoder.onnx'),opts,providers=['CPUExecutionProvider'])
        assert self.enc.get_inputs()[0].shape==[1,1762]
        assert self.dec.get_inputs()[0].shape==[1,994]
    def state(self,d,last):
        R=Rotation.from_quat(d.qpos[[4,5,6,3]]).as_matrix()
        return [d.qvel[3:6].copy(),(d.qpos[7:]-Q0)[MJ_TO_IL],d.qvel[6:][MJ_TO_IL],last.copy(),R.T@np.array([0,0,-1.])]
    def reset(self,d):
        s=self.state(d,np.zeros(29));self.hist=[deque([v.copy() for _ in range(10)],maxlen=10) for v in s]
    def update(self,d,act):
        for hist,v in zip(self.hist,self.state(d,act)):hist.append(v.copy())
    def act(self,d,q,dq,quat,i):
        ids=np.minimum(i+np.arange(10)*5,len(q)-1)
        refR=Rotation.from_quat(quat[ids][:,[1,2,3,0]]).as_matrix()
        robotR=Rotation.from_quat(d.qpos[[4,5,6,3]]).as_matrix()
        relative=robotR.T@refR
        # Zero unused encoder modes; mode ID 0 is scalar zero, not one-hot.
        enc=np.zeros(1762,dtype=np.float32)
        enc[4:294]=q[ids][:,MJ_TO_IL].ravel()
        enc[294:584]=dq[ids][:,MJ_TO_IL].ravel()
        enc[601:661]=relative[:,:,:2].ravel()
        token=self.enc.run(None,{'obs_dict':enc[None]})[0].reshape(-1)
        obs=np.concatenate([token]+[np.array(h).ravel() for h in self.hist]).astype(np.float32)
        assert obs.shape==(994,)
        raw=self.dec.run(None,{'obs_dict':obs[None]})[0].reshape(29)
        assert np.isfinite(raw).all()
        raw=np.clip(raw,-20,20)
        return Q0+raw[IL_TO_MJ]*SCALE,raw

def simulate(m,policy,q,quat,root,perturb_seed=None):
    d=mujoco.MjData(m);d.qpos[:3]=root[0];d.qpos[3:7]=quat[0];d.qpos[7:]=q[0]
    floor_align(m,d)
    if perturb_seed is not None:
        rng=np.random.default_rng(perturb_seed);d.qvel[:2]=rng.uniform(-.03,.03,2);d.qvel[3:6]=rng.uniform(-.03,.03,3)
    policy.reset(d); dq=np.gradient(q,.02,axis=0);states=[];contact=[];fall=None
    for i in range(len(q)):
        target,act=policy.act(d,q,dq,quat,i)
        for _ in range(10):
            d.ctrl[:]=np.clip(KP*(target-d.qpos[7:])-KD*d.qvel[6:],-EFF,EFF)
            mujoco.mj_step(m,d)
        mujoco.mj_forward(m,d);policy.update(d,act)
        states.append(d.qpos.copy());contact.append(d.ncon)
        tilt=np.linalg.norm(Rotation.from_quat(d.qpos[[4,5,6,3]]).as_euler('xyz')[:2])
        if fall is None and (d.qpos[2]<.35 or tilt>np.pi/3):fall=(i+1)*.02
    return np.array(states),contact,fall

if __name__=='__main__':
    import json
    OUT.mkdir(parents=True,exist_ok=True);m=load_model();p=Policy()
    q=np.repeat(Q0[None],300,axis=0);quat=np.tile([1.,0,0,0],(300,1));root=np.tile([0.,0,.8],(300,1))
    s,c,f=simulate(m,p,q,quat,root)
    r=dict(test='6 second free-base G1 standing gate',fall=f,final_root=s[-1,:3].tolist(),min_height=float(s[:,2].min()))
    (OUT/'standing_gate.json').write_text(json.dumps(r,indent=2));np.savez_compressed(OUT/'standing_gate.npz',qpos=s)
    print(r,flush=True)
