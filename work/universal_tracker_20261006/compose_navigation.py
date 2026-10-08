"""Assemble generated relative commands through uniform GMR and rigid placement."""
import sys,json,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/universal_tracker_20261006';sys.path.insert(0,str(R/'work'));import gmr_probe_20261003 as g
g.M=g.tr.rt.load_model();g.P=None;out=D/'navigation_sequences';out.mkdir(exist_ok=False);rows=json.loads((D/'navigation_commands/manifest.json').read_text());converted={}
def retime(q,n):
 at=np.linspace(0,len(q)-1,n);z=np.stack([np.interp(at,np.arange(len(q)),v) for v in q.T],1);z[:,3:7]=Slerp(np.arange(len(q)),Rotation.from_quat(q[:,[4,5,6,3]]))(at).as_quat()[:,[3,0,1,2]];return z

def place(q,anchor):
 q=q.copy();yaw=Rotation.from_quat(anchor[[4,5,6,3]]).as_euler('xyz')[2]-Rotation.from_quat(q[0,[4,5,6,3]]).as_euler('xyz')[2];rot=Rotation.from_euler('z',yaw);origin=q[0,:3].copy();origin[2]=0;target=anchor[:3].copy();target[2]=0;q[:,:3]=rot.apply(q[:,:3]-origin)+target;q[:,3:7]=(rot*Rotation.from_quat(q[:,[4,5,6,3]])).as_quat()[:,[3,0,1,2]];return q

def blend(q0,q1,n):
 u=np.linspace(0,1,n+1)[1:];u=u*u*(3-2*u);q=q0[None]+u[:,None]*(q1-q0)[None];q[:,3:7]=Slerp([0,1],Rotation.from_quat(np.array([q0,q1])[:,[4,5,6,3]]))(u).as_quat()[:,[3,0,1,2]];return q
for row in rows:
 q=g.convert(np.load(row['path']),'uniform')
 if row['task']=='walk':q=retime(q,round((row['command']/.45+.8)*20))
 converted[(row['route'],row['stage'])]=q
manifest=[]
for route in sorted({r['route'] for r in rows}):
 stages=sorted([r for r in rows if r['route']==route],key=lambda r:r['stage']);sequence=None;segments=[];anchors=[]
 for row in stages:
  q=converted[(route,row['stage'])]
  if sequence is not None:
   q=place(q,sequence[-1]);anchors.append(round(len(sequence)*2.5)+50);sequence=np.r_[sequence,blend(sequence[-1],q[0],20)];start=len(sequence);sequence=np.r_[sequence,q]
  else:start=0;sequence=q.copy()
  end=len(sequence);segments.append(dict(row,start20=start,end20=end,start50=round(start*2.5)+50,end50=round((end-1)*2.5)+50,reference_displacement_m=float(np.linalg.norm(q[-1,:2]-q[0,:2]))))
 stand=sequence[-1].copy();stand[7:]=g.tr.rt.Q0;stand[3:7]=Rotation.from_euler('z',Rotation.from_quat(stand[[4,5,6,3]]).as_euler('xyz')[2]).as_quat()[[3,0,1,2]];data=g.tr.mujoco.MjData(g.M);data.qpos[:]=stand;g.tr.rt.floor_align(g.M,data);stand[2]=data.qpos[2];sequence=np.r_[sequence,blend(sequence[-1],stand,30),np.repeat(stand[None],30,axis=0)]
 path=out/f'route_{route}.npz';np.savez_compressed(path,reference_qpos=sequence,fps=20.);manifest.append(dict(task='navigation',source=f'route_{route}',seed=98061000+route*100,command=None,path=str(path),reference_path=str(path),reference_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),source_type='assembled_FrankenMotion_generated_commands_uniform_GMR',split='development',frames=len(sequence),segments=segments,anchor_frames50=anchors,scope='Relative distance/direction, turn, wave commands. Whole generated walk clips retimed as in prior stable demo; rigid placement, 1s inter-clip blends and final nominal stance. Runtime anchors change future world-frame references only. Per-command errors are measured before subsequent anchoring.'))
(out/'reference_manifest.json').write_text(json.dumps(manifest,indent=2));(out/'manifest.json').write_text(json.dumps(manifest,indent=2));print('Routes assembled',[r['frames']/20 for r in manifest])
