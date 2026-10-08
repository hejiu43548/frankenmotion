"""Matched generation-height diagnostic; frozen wrist-weighted GMR and scene.
Only height input to the learned generator varies. Never rewrite generated joints.
"""
import sys,json,shutil,hashlib
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/universal_tracker_20261006';sys.path.insert(0,str(R/'work'))
import unified_retarget_20261004 as rt
from fk_conversion import convert
rt.init();Original=rt.g.GeneralMotionRetargeting
class FrozenHandGMR(Original):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  for table in [self.ik_match_table1,self.ik_match_table2]:
   for side in ['left','right']:table[side+'_wrist_yaw_link'][1]=80;table[side+'_wrist_yaw_link'][2]=5
  self.setup_retarget_configuration()
rt.g.GeneralMotionRetargeting=FrozenHandGMR
def place(q,anchor):
 q=q.copy();yaw=Rotation.from_quat(anchor[[4,5,6,3]]).as_euler('xyz')[2]-Rotation.from_quat(q[0,[4,5,6,3]]).as_euler('xyz')[2];rot=Rotation.from_euler('z',yaw);origin=q[0,:3].copy();origin[2]=0;target=anchor[:3].copy();target[2]=0;q[:,:3]=rot.apply(q[:,:3]-origin)+target;q[:,3:7]=(rot*Rotation.from_quat(q[:,[4,5,6,3]])).as_quat()[:,[3,0,1,2]];return q
def blend(q0,q1,n):
 u=np.linspace(0,1,n+1)[1:];u=u*u*(3-2*u);q=q0[None]+u[:,None]*(q1-q0)[None];q[:,3:7]=Slerp([0,1],Rotation.from_quat(np.array([q0,q1])[:,[4,5,6,3]]))(u).as_quat()[:,[3,0,1,2]];return q
out=D/'reach_height_scenes';out.mkdir(exist_ok=False);records=[]
for index,row in enumerate(json.loads((D/'reach_height_probe/manifest.json').read_text())):
 original_index=(row['seed']-86005000)*2+int(row['command']>.4);old=R/f'outputs_amass/turn_demo_20261005/development_v4/scene_{original_index:03d}';base=D/f'table_evaluation/baseline_stable/scene_{original_index:03d}';meta=json.loads((base/'reference_contact.json').read_text());segments=meta['segments'];q=np.load(old/'reference_contact.npz')['reference_qpos'].copy();start,end=segments['reach_hold'];transition=segments['transition_to_reach'][0]
 reach=place(rt.g.convert(np.load(row['path']),'uniform'),q[transition-1]);assert len(reach)==end-start;q[start:end]=reach;q[transition:start]=blend(q[transition-1],reach[0],start-transition);turn=segments['turn'][0];q[end:turn]=blend(reach[-1],q[turn],turn-end)
 dest=out/f'scene_{index:03d}';dest.mkdir();inp=dest/'reference_input';inp.mkdir();shutil.copy2(base/'reference_input/scene.mjb',inp/'scene.mjb');shutil.copy2(base/'reference_input/inference_contract.json',inp/'inference_contract.json');np.savez_compressed(inp/'motion.npz',fps=50.,**convert(q,entry=False));np.savez_compressed(dest/'reference_contact.npz',reference_qpos=q,fps=20.)
 original=json.loads((base/'reference_input/result.json').read_text());scene=dict(original['scene'],index=index,source=str(dest),height_command_robot_m=row['height_command_robot_m'],height_probe=row);original['scene']=scene;(inp/'result.json').write_text(json.dumps(original,indent=2));meta['scope']='Development height-conditioning diagnostic. Existing reach adapter, same seed, frozen wrist-weighted GMR80/5. All other full-sequence segments fixed; inter-clip blends recomputed. No scene IK or generated joint editing.';(dest/'reference_contact.json').write_text(json.dumps(meta,indent=2));(dest/'scene.json').write_text(json.dumps(scene,indent=2));records.append(scene);(out/'manifest.json').write_text(json.dumps(records,indent=2));print(index,row['command'],row['height_command_robot_m'],flush=True)
