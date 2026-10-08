"""Geometry and units checks for new shared rewards (CPU, no simulation)."""
import sys
from pathlib import Path
from types import SimpleNamespace as NS
sys.path.insert(0,str(Path(__file__).parent))
import torch
from adaptive_rewards import reference_relative_target_rate,pelvis_relative_distal_position
from mjlab.utils.lab_api.math import quat_from_euler_xyz,matrix_from_quat
names=['pelvis','left_ankle_roll_link','right_ankle_roll_link','left_wrist_yaw_link','right_wrist_yaw_link'];torch.manual_seed(6106);pos=torch.randn(3,5,3);pos[:,0]=0;quat=torch.zeros(3,5,4);quat[...,0]=1
term=NS(cfg=NS(body_names=names),body_pos_w=pos,robot_body_pos_w=pos.clone(),body_quat_w=quat,robot_body_quat_w=quat.clone(),joint_vel=torch.ones(3,29))
env=NS(command_manager=NS(get_term=lambda _:term));assert torch.allclose(pelvis_relative_distal_position(env),torch.ones(3))
# Common translations and rotations must not change pelvis-frame pose accuracy.
yaw=quat_from_euler_xyz(torch.zeros(3),torch.zeros(3),torch.tensor([.3,-.7,1.4]));rot=matrix_from_quat(yaw);term.robot_body_pos_w=torch.einsum('bij,bkj->bki',rot,pos)+torch.tensor([3.,-2.,1.]);term.robot_body_quat_w=yaw[:,None].expand(-1,5,-1)
assert torch.allclose(pelvis_relative_distal_position(env),torch.ones(3),atol=1e-6)
# A local wrist error must reduce the reward even if global placement is arbitrary.
term.robot_body_pos_w[:,3,2]+=.2;assert bool((pelvis_relative_distal_position(env)<.9).all())
scale=torch.linspace(.0745,.5475,29);delta=.02/scale;env.action_manager=NS(action=delta[None].expand(3,-1),prev_action=torch.zeros(3,29),get_term=lambda _:NS(scale=scale));env.step_dt=.02;env.episode_length_buf=torch.tensor([0,1,2])
penalty=reference_relative_target_rate(env);assert penalty[:2].sum()==0;expected=.125*(delta**2).sum();assert torch.allclose(penalty[2],expected)
# Opposing the desired reference motion receives a larger penalty.
env.action_manager.action=-env.action_manager.action;assert reference_relative_target_rate(env)[2]>penalty[2]
print('Passed: rigid-frame invariance, local-pose sensitivity, physical target velocity, episode reset masking')
