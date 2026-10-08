"""Causal SE(2) placement of a generated command in the measured robot frame.

Only future world-frame reference positions/orientations/velocities change.
Generated joint angles, joint velocities, timing, and physical state are untouched.
"""
import numpy as np
from scipy.spatial.transform import Rotation
def anchor_reference(ref,phase,qpos,stage):
 current=Rotation.from_quat(qpos[[4,5,6,3]]).as_euler('xyz')[2]
 desired=Rotation.from_quat(ref['body_quat_w'][phase,0,[1,2,3,0]]).as_euler('xyz')[2]
 yaw=float(np.arctan2(np.sin(current-desired),np.cos(current-desired)));rot=Rotation.from_euler('z',yaw);origin=ref['body_pos_w'][phase,0].copy();origin[2]=0;target=qpos[:3].copy();target[2]=0
 pos=ref['body_pos_w'][phase:];pos[:]=rot.apply((pos-origin).reshape(-1,3)).reshape(pos.shape)+target
 qs=ref['body_quat_w'][phase:];xyzw=qs[:,:,[1,2,3,0]].reshape(-1,4);qs[:]=(rot*Rotation.from_quat(xyzw)).as_quat()[:,[3,0,1,2]].reshape(qs.shape)
 for key in ['body_lin_vel_w','body_ang_vel_w']:
  v=ref[key][phase:];v[:]=rot.apply(v.reshape(-1,3)).reshape(v.shape)
 return dict(stage=stage,phase=int(phase),time_s=phase*.02,measured_root=qpos[:3].tolist(),measured_yaw_rad=float(current),reference_yaw_rad=float(desired),applied_yaw_rad=yaw,reference_root_origin=origin.tolist(),scope='One causal rigid XY/yaw placement at command start. No physical state writes, no joint-angle edits, no action smoothing or tracker switching.')
