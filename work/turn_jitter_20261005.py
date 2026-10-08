"""Paired-reference diagnostic; high-frequency energy is not a naturalness score."""
import argparse,json
from pathlib import Path
import numpy as np
from scipy.signal import butter,sosfiltfilt
import mujoco
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--segment',default='walk');a=p.parse_args();run=Path(a.run)
z=np.load(run/'actual.npz');meta=json.loads((run.parent/'reference_contact.json').read_text());lo,hi=meta['segments'][a.segment];n=len(z['qpos']);ph=z['phases'][:n];idx=np.flatnonzero((ph>=lo*2.5+20)&(ph<hi*2.5-20))
m=mujoco.MjModel.from_binary_path(str(run/'scene.mjb'));d=mujoco.MjData(m);head=[];torso=[]
bid=m.body('robot/torso_link').id
for i in idx:
 d.qpos[:]=z['qpos'][i];d.qvel[:]=z['qvel'][i];mujoco.mj_forward(m,d);v=np.zeros(6);mujoco.mj_objectVelocity(m,d,mujoco.mjtObj.mjOBJ_BODY,bid,v,0);torso.append(v.copy())
sos=butter(4,5,fs=50,btype='highpass',output='sos')
def stats(v):
 v=np.asarray(v);hf=sosfiltfilt(sos,v,axis=0)
 return dict(rms=float(np.sqrt(np.mean(v*v))),highpass_5hz_rms=float(np.sqrt(np.mean(hf*hf))),difference_rms=float(np.sqrt(np.mean(np.diff(v,axis=0)**2))))
report=dict(run=str(run),frames=len(idx),trim_seconds=.4,torso_angular_velocity=stats(np.asarray(torso)[:,:3]),torso_linear_velocity=stats(np.asarray(torso)[:,3:]),joint_velocity=stats(z['qvel'][idx,6:]),action=stats(z['actions'][idx]),scope='Central approach phase, 5Hz fourth-order zero-phase high-pass; compare identical reference and initialization. No trajectory modification.')
(run/(a.segment+'_jitter_metrics.json')).write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
