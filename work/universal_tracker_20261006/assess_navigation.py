"""Measure each relative command before the following reference anchoring."""
import argparse,json,sys
from pathlib import Path
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
R=Path('/home/pku/frankenmotion');sys.path[:0]=[str(R/'work'),str(R/'outputs_amass/franken_eleven_20261003/code')]
import transfer as tr
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();root=Path(a.run)
source=tr.rt.load_model();height=tr.robot_height(source);rows=[]
def yaw(q):return np.unwrap(Rotation.from_quat(q[:,[4,5,6,3]]).as_euler('xyz')[:,2])
for record in json.loads((root/'results.json').read_text()):
 d=Path(record['run']);actual=np.load(d/'actual.npz');ref=np.load(d/'motion.npz');model=mujoco.MjModel.from_binary_path(str(d/'scene.mjb'))
 order=[int(model.jnt_qposadr[model.joint('robot/'+source.joint(i).name).id]) for i in range(1,source.njnt)]
 states=np.c_[actual['qpos'][:,:7],actual['qpos'][:,order]];refstates=np.c_[ref['body_pos_w'][:,0],ref['body_quat_w'][:,0]]
 for s in record['segments']:
  first,last=s['start50'],s['end50'];row={k:s[k] for k in ['route','stage','task','command','direction_rad']};row['complete']=len(states)>last
  if row['complete']:
   for key,q in [('actual',states),('g1_reference',refstates)]:
    angle=yaw(q[first:last+1]);delta=q[last,:2]-q[first,:2]
    if s['task']=='walk':
     direction=float((np.arctan2(delta[1],delta[0])-angle[0]+np.pi)%(2*np.pi)-np.pi)
     row[key]=dict(distance_m=float(np.linalg.norm(delta)),direction_rad=direction,distance_error_m=float(abs(np.linalg.norm(delta)-s['command'])),direction_error_deg=float(abs((direction-s['direction_rad']+np.pi)%(2*np.pi)-np.pi)*180/np.pi))
    elif s['task']=='turn':
     turn=float(-np.degrees(angle[-1]-angle[0]));row[key]=dict(right_turn_deg=turn,error_deg=abs(turn-s['command']))
   if s['task']=='wave':
    positions=tr.get_positions(source,states[first:last+2]);times=np.arange(len(positions))*.02;want=np.arange(120)*.05;assert times[-1]>=want[-1];positions=np.stack([np.interp(want,times,v) for v in positions.reshape(len(positions),-1).T],1).reshape(-1,24,3)
    row['actual']=tr.measure(positions,'wave',height)
  rows.append(row)
out=dict(scope='Stage endpoints before next command anchoring. Walk directions relative to each stage initial pelvis yaw. No simulator reset or endpoint alignment in measurements. Wave uses frozen benchmark metric.',results=rows)
(root/'command_metrics.json').write_text(json.dumps(out,indent=2,default=lambda x:x.item()));print(json.dumps(out,indent=2,default=lambda x:x.item()))
