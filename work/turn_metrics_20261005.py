"""Full turn/departure success, in addition to prior contact/reach requirements."""
import json,argparse,subprocess,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=Path(a.run)
subprocess.run([sys.executable,str(Path(__file__).with_name('reach_metrics_20261005.py')),'--run',str(run)],check=True,stdout=subprocess.DEVNULL)
r=json.loads((run/'command_metrics.json').read_text());result=json.loads((run/'result.json').read_text());scene=result['scene'];meta=json.loads((run.parent/'reference_contact.json').read_text());z=np.load(run/'actual.npz');n=len(z['qpos']);phase=z['phases'][:n];yaw=np.unwrap(Rotation.from_quat(z['qpos'][:,[4,5,6,3]]).as_euler('xyz')[:,2]);root=z['qpos'][:,:2]
def ids(name):
 lo,hi=meta['segments'][name];return np.flatnonzero((phase>=lo*2.5)&(phase<hi*2.5))
ti=ids('turn');wi=ids('away');turn=float(-np.degrees(yaw[ti[-1]]-yaw[ti[0]])) if len(ti)>1 else None;err=abs(turn-scene['turn_command_deg']) if turn is not None else None;distance=float(np.linalg.norm(root[wi[-1]]-root[wi[0]])) if len(wi)>1 else None;enderr=float(np.linalg.norm(root[-1]-scene['reference_final_xy']));distanceerr=abs(distance-scene['away_command_robot_m']) if distance is not None else None
events=json.loads((run/'anchor_events.json').read_text()) if (run/'anchor_events.json').exists() else []
for event in events:
 if event['stage']=='turn' and len(ti)>1:
  delta=yaw[ti[-1]]-event['measured_yaw_rad'];turn=float(-np.degrees(np.arctan2(np.sin(delta),np.cos(delta))));err=abs(turn-scene['turn_command_deg'])
 if event['stage']=='away' and len(wi)>1:
  distance=float(np.linalg.norm(root[wi[-1]]-np.asarray(event['measured_root'])[:2]));distanceerr=abs(distance-scene['away_command_robot_m'])
if events:enderr=float(np.linalg.norm(root[-1]-np.load(run/'motion.npz')['body_pos_w'][-1,0,:2]))
r.update(reach_interaction_success=r['success'],turn_command_deg=scene['turn_command_deg'],actual_turn_deg=turn,turn_error_deg=err,turn_pass=err is not None and err<=12,away_command_robot_m=scene['away_command_robot_m'],actual_away_displacement_m=distance,away_error_m=distanceerr,final_reference_endpoint_error_m=enderr,away_pass=distanceerr is not None and distanceerr<=.25 and enderr<=.3)
r['success']=bool(r['reach_interaction_success'] and r['turn_pass'] and r['away_pass']);r['scope']+=' Full demo additionally requires signed right turn error <=12deg, departure displacement error <=0.25m, final reference endpoint error <=0.3m. Jitter reported separately.';(run/'command_metrics.json').write_text(json.dumps(r,indent=2));print(json.dumps(r,indent=2))
