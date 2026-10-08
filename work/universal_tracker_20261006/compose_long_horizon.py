"""Post-test exploratory four-cycle composition, fixed route_0; no policy or command retuning."""
import json,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/universal_tracker_20261006';out=D/'long_horizon_sequence';out.mkdir(exist_ok=False)
row=json.loads((D/'navigation_sequences/manifest.json').read_text())[0];original=np.load(row['reference_path'])['reference_qpos'];sequence=None;segments=[];anchors=[]
def place(q,anchor):
 q=q.copy();rot=Rotation.from_euler('z',Rotation.from_quat(anchor[[4,5,6,3]]).as_euler('xyz')[2]-Rotation.from_quat(q[0,[4,5,6,3]]).as_euler('xyz')[2]);origin=q[0,:3].copy();origin[2]=0;target=anchor[:3].copy();target[2]=0;q[:,:3]=rot.apply(q[:,:3]-origin)+target;q[:,3:7]=(rot*Rotation.from_quat(q[:,[4,5,6,3]])).as_quat()[:,[3,0,1,2]];return q

def blend(q0,q1,n=20):
 u=np.linspace(0,1,n+1)[1:];u=u*u*(3-2*u);q=q0[None]+u[:,None]*(q1-q0)[None];q[:,3:7]=Slerp([0,1],Rotation.from_quat(np.array([q0,q1])[:,[4,5,6,3]]))(u).as_quat()[:,[3,0,1,2]];return q
for cycle in range(4):
 for old in row['segments']:
  q=original[old['start20']:old['end20']].copy()
  if sequence is not None:
   q=place(q,sequence[-1]);anchors.append(round(len(sequence)*2.5)+50);sequence=np.r_[sequence,blend(sequence[-1],q[0])]
  start=0 if sequence is None else len(sequence);sequence=q.copy() if sequence is None else np.r_[sequence,q];end=len(sequence)
  segments.append(dict(old,stage=len(segments),cycle=cycle,start20=start,end20=end,start50=round(start*2.5)+50,end50=round((end-1)*2.5)+50))
# Retain original final stand transition rigidly placed after the final wave.
tail=original[row['segments'][-1]['end20']-1:];tail=place(tail,sequence[-1]);sequence=np.r_[sequence,tail[1:]]
path=out/'four_cycles.npz';np.savez_compressed(path,reference_qpos=sequence,fps=20.)
r=dict(row,task='navigation',source='four_cycles',seed=98061111,path=str(path),reference_path=str(path),reference_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),frames=len(sequence),segments=segments,anchor_frames50=anchors,scope=__doc__+' Reuse the same six generated clips four times with 1s transitions. Runtime reference anchoring only; actual simulator state never reset after initialization. Repetition is not 24 independent samples and not fresh generation. Relative commands, not absolute closed-loop global navigation.')
(out/'manifest.json').write_text(json.dumps([r],indent=2));print('duration',len(sequence)/20,'stages',len(segments))
