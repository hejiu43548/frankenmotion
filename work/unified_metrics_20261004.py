"""Recompute source metrics; a cached training proxy is optional metadata only."""
import numpy as np
import transfer as tr
from audit_results import canonical

def native_metrics(row):
 z=np.load(row['path']);p=canonical(z['joints_zup_m']);h=float(z['human_height']);r=tr.measure(p,row['task'],h)
 torso=(p[:,16]+p[:,17])/2-p[:,0];tilt=np.rad2deg(np.arctan2(np.linalg.norm(torso[:,:2],axis=-1),torso[:,2]));vel=np.linalg.norm(np.diff(p[:,0],axis=0),axis=-1)*20
 r.update(boundary_torso_over45=bool(np.any(np.r_[tilt[:5],tilt[-5:]]>45)),root_speed_peak_human_equivalent=float(vel.max()*tr.HH/h),cached_training_quantity=float(z['human_quantity']) if 'human_quantity' in z else None)
 return r
