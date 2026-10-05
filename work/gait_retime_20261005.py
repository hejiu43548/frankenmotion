"""Retime walk reference only; preserve spatial trajectory and original for comparison."""
import argparse,json,numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation,Slerp
p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args()
for row in json.loads((Path(a.folder)/'manifest.json').read_text()):
 s=Path(row['source']);meta=json.loads((s/'reference_contact.json').read_text());assert 'temporal_retiming' not in meta
 (s/'reference_slow.json').write_text(json.dumps(meta,indent=2));(s/'reference_slow.npz').write_bytes((s/'reference_contact.npz').read_bytes())
 q=np.load(s/'reference_contact.npz')['reference_qpos'];lo,hi=meta['segments']['walk'];walk=q[lo:hi];duration=float(np.clip(row['distance_robot_m']/.45+.8,2.8,4.8));n=round(duration*20);t=np.linspace(0,len(walk)-1,n);new=np.stack([np.interp(t,np.arange(len(walk)),v) for v in walk.T],1);new[:,3:7]=Slerp(np.arange(len(walk)),Rotation.from_quat(walk[:,[4,5,6,3]]))(t).as_quat()[:,[3,0,1,2]];q=np.r_[q[:lo],new,q[hi:]];delta=n-(hi-lo);meta['segments']['walk']=[lo,lo+n]
 for k in ['settle','reach_hold']:meta['segments'][k]=[v+delta for v in meta['segments'][k]]
 meta['reference_frames']=len(q);meta['temporal_retiming']=dict(original_frames=hi-lo,new_frames=n,walk_duration_s=duration,spatial_path_unchanged=True,rule='clip(goal_distance/0.45+0.8,2.8,4.8) seconds; interpolation and rotation slerp')
 np.savez_compressed(s/'reference_contact.npz',reference_qpos=q,fps=20.);(s/'reference_contact.json').write_text(json.dumps(meta,indent=2))
