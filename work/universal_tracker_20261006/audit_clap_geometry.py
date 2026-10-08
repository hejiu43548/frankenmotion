"""Descriptive wrist-approach events for text demos; not validated semantic success."""
import json,sys,argparse
from pathlib import Path
import numpy as np,mujoco
from scipy.signal import find_peaks
R=Path('/home/pku/frankenmotion');sys.path[:0]=[str(R/'work'),str(R/'outputs_amass/franken_eleven_20261003/code')]
import transfer as tr
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();root=Path(a.run);source=tr.rt.load_model();height=tr.robot_height(source)
def measure(pos,h,fps):
 scale=tr.HH/h;distance=np.linalg.norm(pos[:,20]-pos[:,21],axis=1)*scale;wrist_height=((pos[:,20,2]+pos[:,21,2])/2-pos[:,0,2])*scale
 peaks,_=find_peaks(-distance,prominence=.05,distance=max(1,int(.3*fps)))
 events=[int(i) for i in peaks if distance[i]<.18 and wrist_height[i]>.15]
 return dict(min_wrist_distance_human_m=float(distance.min()),near_approach_count=len(events),approach_times_s=[i/fps for i in events],wrist_distance_human_m=distance.tolist(),wrist_height_above_pelvis_human_m=wrist_height.tolist(),fps=fps)
results=[]
for row in json.loads((root/'results.json').read_text()):
 if row['task']!='clap' or 'error' in row:continue
 z=np.load(row['path']);ref=np.load(row['reference_path'])['reference_qpos'];run=Path(row['run']);m=mujoco.MjModel.from_binary_path(str(run/'scene.mjb'));actual=np.load(run/'actual.npz')['qpos'];order=[int(m.jnt_qposadr[m.joint('robot/'+source.joint(i).name).id]) for i in range(1,source.njnt)];q=np.c_[actual[:,:7],actual[:,order]]
 data=mujoco.MjData(m);left=m.geom('robot/left_hand_collision').id;right=m.geom('robot/right_hand_collision').id;left_region={left,m.geom('robot/left_wrist_collision').id};right_region={right,m.geom('robot/right_wrist_collision').id};contacts=[];region_contacts=[]
 for i,state in enumerate(actual):
  data.qpos[:]=state;mujoco.mj_forward(m,data)
  distances=[float(c.dist) for c in data.contact if {int(c.geom1),int(c.geom2)}=={left,right}]
  if distances:contacts.append(dict(frame=i,time_s=i/50.,minimum_distance_m=min(distances)))
  for c in data.contact:
   if (int(c.geom1) in left_region and int(c.geom2) in right_region) or (int(c.geom2) in left_region and int(c.geom1) in right_region):region_contacts.append(dict(frame=i,time_s=i/50.,geometries=[m.geom(int(c.geom1)).name,m.geom(int(c.geom2)).name],distance_m=float(c.dist)))
 episodes=[]
 for contact in contacts:
  if not episodes or contact['frame']>episodes[-1]['last_frame']+1:episodes.append(dict(first_frame=contact['frame'],last_frame=contact['frame']))
  else:episodes[-1]['last_frame']=contact['frame']
 results.append(dict(source=row['source'],caption=row.get('caption'),complete=row['physical_complete'],human=measure(z['joints_zup_m'],float(z['human_height']),20),reference=measure(tr.get_positions(source,ref),height,20),actual=measure(tr.get_positions(source,q[50:]),height,50),actual_hand_geometry_contact=dict(frames=len(contacts),sampled_duration_s=len(contacts)/50.,episodes=episodes,contacts=contacts,hand_or_wrist_contacts=region_contacts,scope='Recorded poses re-evaluated for collision geometry only; not a contact-force or impulse measurement. Wrist distance does not account for palm thickness. Hand-to-wrist is reported separately and must not be called palm-to-palm clapping.')))
output=dict(scope=__doc__,definition='Local minima of wrist separation with >=5cm prominence and >=0.3s spacing; separation<18cm and mean wrist height>15cm above pelvis, in frozen human-equivalent units. This detects near approaches, not hand-surface contact or verified clapping semantics. Threshold chosen as a descriptive demo diagnostic, not part of11-task scores.',results=results)
(root/'clap_geometry.json').write_text(json.dumps(output,indent=2));print([(r['source'],[(k,r[k]['near_approach_count'],round(r[k]['min_wrist_distance_human_m'],3)) for k in ['human','reference','actual']]) for r in results])
