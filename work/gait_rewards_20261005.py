"""Task-independent foot/knee fidelity rewards; no task or phase routing."""
import torch
from mjlab.utils.lab_api.math import quat_error_magnitude

def feet_position(env,xy_std=.06,z_std=.025):
 t=env.command_manager.get_term('motion');ids=[t.cfg.body_names.index(n) for n in ['left_ankle_roll_link','right_ankle_roll_link']];error=t.body_pos_w[:,ids]-t.robot_body_pos_w[:,ids];scaled=error/error.new_tensor([xy_std,xy_std,z_std]);return torch.exp(-scaled.square().sum(-1)).mean(-1)
def feet_velocity(env,std=.5):
 t=env.command_manager.get_term('motion');ids=[t.cfg.body_names.index(n) for n in ['left_ankle_roll_link','right_ankle_roll_link']];error=t.body_lin_vel_w[:,ids]-t.robot_body_lin_vel_w[:,ids];return torch.exp(-error.square().sum(-1)/(std*std)).mean(-1)
def feet_orientation(env,std=.3):
 t=env.command_manager.get_term('motion');ids=[t.cfg.body_names.index(n) for n in ['left_ankle_roll_link','right_ankle_roll_link']];error=quat_error_magnitude(t.body_quat_w[:,ids],t.robot_body_quat_w[:,ids]);return torch.exp(-error.square()/(std*std)).mean(-1)
def knee_position(env,std=.15):
 t=env.command_manager.get_term('motion');names=list(env.scene['robot'].joint_names);ids=[names.index(n) for n in ['left_knee_joint','right_knee_joint']];error=t.joint_pos[:,ids]-t.robot_joint_pos[:,ids];return torch.exp(-error.square()/(std*std)).mean(-1)

def sole_data(env):
 if not hasattr(env,'_gait_sole_points'):
  import numpy as np
  from scipy.spatial.transform import Rotation
  m=env.sim.mj_model;points=[];radii=[]
  for side in ['left','right']:
   ps=[];rs=[]
   for g in range(m.ngeom):
    if m.geom(g).name.startswith('robot/'+side+'_foot') and 'collision' in m.geom(g).name:
     assert int(m.geom_type[g])==3
     rot=Rotation.from_quat(m.geom_quat[g][[1,2,3,0]]).as_matrix();axis=rot[:,2]*m.geom_size[g,1]
     ps.extend([m.geom_pos[g]+axis,m.geom_pos[g]-axis]);rs.extend([m.geom_size[g,0]]*2)
   points.append(ps);radii.append(rs)
  env._gait_sole_points=(torch.tensor(np.array(points),dtype=torch.float32,device=env.device),torch.tensor(np.array(radii),dtype=torch.float32,device=env.device))
 from mjlab.utils.lab_api.math import quat_apply
 t=env.command_manager.get_term('motion');ids=[t.cfg.body_names.index(n) for n in ['left_ankle_roll_link','right_ankle_roll_link']];points,radii=env._gait_sole_points;batch=env.num_envs;points=points[None].expand(batch,-1,-1,-1);shape=points.shape
 def calc(pos,quat):
  rotated=quat_apply(quat[:,:,None].expand(-1,-1,shape[2],-1).reshape(-1,4),points.reshape(-1,3)).reshape(shape);clear=(pos[:,:,None,2]+rotated[:,:,:,2]-radii[None]);values,idx=clear.min(-1);arm=rotated.gather(2,idx[:,:,None,None].expand(-1,-1,1,3)).squeeze(2);return values,arm
 target,_=calc(t.body_pos_w[:,ids],t.body_quat_w[:,ids]);actual,arm=calc(t.robot_body_pos_w[:,ids],t.robot_body_quat_w[:,ids]);return target,actual,arm,ids

def feet_clearance(env,std=.015):
 target,actual,_,_=sole_data(env);weights=1.+(target>.015).float();return (torch.exp(-(actual-target).square()/(std*std))*weights).sum(-1)/weights.sum(-1)
def stance_velocity(env,std=.12):
 target,actual,arm,ids=sole_data(env);t=env.command_manager.get_term('motion');vel=t.robot_body_lin_vel_w[:,ids]+torch.cross(t.robot_body_ang_vel_w[:,ids],arm,dim=-1);score=torch.exp(-vel[:,:,:2].square().sum(-1)/(std*std));return torch.where(target<.008,score,torch.ones_like(score)).mean(-1)
