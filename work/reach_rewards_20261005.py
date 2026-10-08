"""Reference-derived palm tracking rewards for every clip, without task IDs or routing."""
import torch
from mjlab.utils.lab_api.math import quat_apply,quat_error_magnitude

def right_palm_position(env,xy_std=.04,z_std=.02):
 t=env.command_manager.get_term('motion');idx=t.cfg.body_names.index('right_wrist_yaw_link');offset=t.body_pos_w.new_tensor([.11,.01,0.]).expand(env.num_envs,-1);target=t.body_pos_w[:,idx]+quat_apply(t.body_quat_w[:,idx],offset);actual=t.robot_body_pos_w[:,idx]+quat_apply(t.robot_body_quat_w[:,idx],offset);err=(actual-target)/target.new_tensor([xy_std,xy_std,z_std]);return torch.exp(-err.square().sum(-1))
def right_palm_orientation(env,std=.35):
 t=env.command_manager.get_term('motion');idx=t.cfg.body_names.index('right_wrist_yaw_link');err=quat_error_magnitude(t.body_quat_w[:,idx],t.robot_body_quat_w[:,idx]);return torch.exp(-err.square()/(std*std))

def right_palm_dense_error(env,xy_std=.05,z_std=.025):
 t=env.command_manager.get_term('motion');idx=t.cfg.body_names.index('right_wrist_yaw_link');offset=t.body_pos_w.new_tensor([.11,.01,0.]).expand(env.num_envs,-1);target=t.body_pos_w[:,idx]+quat_apply(t.body_quat_w[:,idx],offset);actual=t.robot_body_pos_w[:,idx]+quat_apply(t.robot_body_quat_w[:,idx],offset);error=(actual-target)/target.new_tensor([xy_std,xy_std,z_std]);return -torch.linalg.vector_norm(error,dim=-1).clamp(max=4.)

def right_wrist_relative_xy(env,std=.03):
 t=env.command_manager.get_term('motion');w=t.cfg.body_names.index('right_wrist_yaw_link');root=t.cfg.body_names.index('pelvis');ref=t.body_pos_w[:,w,:2]-t.body_pos_w[:,root,:2];actual=t.robot_body_pos_w[:,w,:2]-t.robot_body_pos_w[:,root,:2];return -torch.linalg.vector_norm((actual-ref)/std,dim=-1).clamp(max=4.)

def feet_clearance_dense(env,std=.02):
 from gait_rewards_20261005 import sole_data
 target,actual,_,_=sole_data(env);weights=1.+(target>.015).float();error=((actual-target).abs()/std).clamp(max=3.);return -(error*weights).sum(-1)/weights.sum(-1)
