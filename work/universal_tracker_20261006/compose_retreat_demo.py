"""A retreat sequence from existing generated numeric commands and text-only point.
Walk, point, walk backwards, sidestep right, turn, and leave.
No tracker choice depends on this demo; evaluate only the globally frozen policy.
"""
import sys,json,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/universal_tracker_20261006';sys.path.insert(0,str(R/'work'));import gmr_probe_20261003 as g
from fk_conversion import convert
g.M=g.tr.rt.load_model();out=D/'retreat_sequences';out.mkdir(exist_ok=False);motions=D/'retreat_motion';motions.mkdir(exist_ok=False)
def place(q,anchor):
 q=q.copy();yaw=Rotation.from_quat(anchor[[4,5,6,3]]).as_euler('xyz')[2]-Rotation.from_quat(q[0,[4,5,6,3]]).as_euler('xyz')[2];rot=Rotation.from_euler('z',yaw);origin=q[0,:3].copy();origin[2]=0;target=anchor[:3].copy();target[2]=0;q[:,:3]=rot.apply(q[:,:3]-origin)+target;q[:,3:7]=(rot*Rotation.from_quat(q[:,[4,5,6,3]])).as_quat()[:,[3,0,1,2]];return q

def blend(q0,q1,n):
 u=np.linspace(0,1,n+1)[1:];u=u*u*(3-2*u);q=q0[None]+u[:,None]*(q1-q0)[None];q[:,3:7]=Slerp([0,1],Rotation.from_quat(np.array([q0,q1])[:,[4,5,6,3]]))(u).as_quat()[:,[3,0,1,2]];return q
nav=json.loads((D/'navigation_sequences/manifest.json').read_text());points=json.loads((D/'generated_point_corrected_motion/manifest.json').read_text());official=json.loads((D/'generated_official_motion/manifest.json').read_text());bow=next(r for r in official if r['source']=='bend_09156_s1');records=[]
fresh=json.loads((D/'fresh_final/native_manifest.json').read_text());back=next(r for r in fresh if r['task']=='back_walk' and r['source']=='back_walk_p0_s0' and r['command_index']==2);side=next(r for r in fresh if r['task']=='sidestep' and r['source']=='sidestep_p0_s0' and r['command_index']==2)
for route in range(1):
 source=next(r for r in nav if r['source']==f'route_{route}');nq=np.load(source['reference_path'])['reference_qpos'];ns=source['segments'];point=next(r for r in points if r['source']==f'point_01584_s{route}')
 selected=[]
 for task,src in [('walk',ns[0]),('point',point),('back_walk',back),('sidestep',side),('turn',ns[1]),('walk',ns[4])]:
  if task in ['walk','turn']:q=nq[src['start20']:src['end20']].copy();meta=dict(src)
  else:
   q=np.load(src['reference_path'])['reference_qpos'];meta=dict(src,task=task,command=src.get('command'),direction_rad=0.)
   if task=='bow':
    assert len(q)>=144;meta['source_trim20']=[80,144];meta['source_full_frames20']=len(q);meta['trim_reason']='Use first generated bow/return segment, omit caption T-pose and extra turns. Exact source poses retained within time window, then ordinary rigid placement/blending.';q=q[80:144].copy()
  selected.append((q,meta))
 sequence=None;segments=[];anchors=[]
 for stage,(q,meta) in enumerate(selected):
  if sequence is not None:q=place(q,sequence[-1]);anchors.append(round(len(sequence)*2.5)+50);sequence=np.r_[sequence,blend(sequence[-1],q[0],20)];start=len(sequence);sequence=np.r_[sequence,q]
  else:start=0;sequence=q.copy()
  segments.append(dict(meta,route=route,stage=stage,start20=start,end20=len(sequence),start50=round(start*2.5)+50,end50=round((len(sequence)-1)*2.5)+50,reference_displacement_m=float(np.linalg.norm(q[-1,:2]-q[0,:2]))))
 stand=sequence[-1].copy();stand[7:]=g.tr.rt.Q0;stand[3:7]=Rotation.from_euler('z',Rotation.from_quat(stand[[4,5,6,3]]).as_euler('xyz')[2]).as_quat()[[3,0,1,2]];sd=g.tr.mujoco.MjData(g.M);sd.qpos[:]=stand;g.tr.rt.floor_align(g.M,sd);stand[2]=sd.qpos[2];sequence=np.r_[sequence,blend(sequence[-1],stand,30),np.repeat(stand[None],30,axis=0)];path=out/f'retreat_{route}.npz';np.savez_compressed(path,reference_qpos=sequence,fps=20.);motion=motions/f'retreat_{route}_motion.npz';np.savez_compressed(motion,fps=50.,**convert(sequence));record=dict(task='retreat',source=f'retreat_{route}',seed=None,command=None,path=str(path),reference_path=str(path),reference_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),motion_path=str(motion),motion_sha256=hashlib.sha256(motion.read_bytes()).hexdigest(),source_type='assembled_FrankenMotion_generated_commands_and_official_text',split='presentation_development',frames=len(sequence),segments=segments,anchor_frames50=anchors,scope=__doc__+' Uniform GMR sources; previously declared walk retiming, rigid placement and1s transitions, final nominal stance. Whole source clips preserved, including the numeric backward-speed and right-sidestep commands. No within-clip pose edits. No real push, ball, finger actuation, perception, or human interaction is simulated.');records.append(record);print(record['source'],len(sequence)/20,'s',flush=True)
(out/'manifest.json').write_text(json.dumps(records,indent=2));(motions/'manifest.json').write_text(json.dumps(records,indent=2));(out/'protocol.json').write_text(json.dumps(dict(scope=__doc__,candidates=1,source_selection='Navigation route0, right-arm point seed0, first fresh source/middle command for back_walk and sidestep, already exposed in frozen tests. Post-test presentation composition only; no new generalization score.',requires_single_globally_frozen_policy=True),indent=2))
