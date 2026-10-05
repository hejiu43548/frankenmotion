from pathlib import Path
import json,shutil,numpy as np
from scipy.spatial.transform import Rotation,Slerp
R=Path('/home/pku/frankenmotion');OLD=R/'outputs_amass/table_demo_20261005';D=R/'outputs_amass/gait_demo_20261005';rows=json.loads((OLD/'development_v3/manifest.json').read_text())
for label in ['dev_slow','dev_timed']:
 out=D/label;out.mkdir(exist_ok=False);records=[]
 for row in rows:
  old=Path(row['source']);dest=out/old.name;dest.mkdir();r=dict(row);r['source']=str(dest)
  for name in ['human_walk.npz','human_reach.npz','raw_retargets.npz','reference_contact.npz','reference_contact.json']:shutil.copy2(old/name,dest/name)
  if label=='dev_timed':
   meta=json.loads((dest/'reference_contact.json').read_text());q=np.load(dest/'reference_contact.npz')['reference_qpos'];lo,hi=meta['segments']['walk'];walk=q[lo:hi];duration=float(np.clip(r['distance_robot_m']/.45+.8,2.8,4.8));n=round(duration*20);t=np.linspace(0,len(walk)-1,n);new=np.stack([np.interp(t,np.arange(len(walk)),v) for v in walk.T],1);new[:,3:7]=Slerp(np.arange(len(walk)),Rotation.from_quat(walk[:,[4,5,6,3]]))(t).as_quat()[:,[3,0,1,2]];q=np.r_[q[:lo],new,q[hi:]];delta=n-(hi-lo)
   meta['segments']['walk']=[lo,lo+n]
   for key in ['settle','reach_hold']:meta['segments'][key]=[v+delta for v in meta['segments'][key]]
   meta['reference_frames']=len(q);meta['temporal_retiming']=dict(original_frames=hi-lo,new_frames=n,walk_duration_s=duration,spatial_path_unchanged=True,rule='clip(goal_distance/0.45+0.8,2.8,4.8) seconds; interpolation and rotation slerp')
   np.savez_compressed(dest/'reference_contact.npz',reference_qpos=q,fps=20.);(dest/'reference_contact.json').write_text(json.dumps(meta,indent=2))
  (dest/'scene.json').write_text(json.dumps(r,indent=2));records.append(r)
 (out/'manifest.json').write_text(json.dumps(records,indent=2))
print('Copied six development layouts, unchanged and retimed references')
