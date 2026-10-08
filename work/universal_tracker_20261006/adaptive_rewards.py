"""Task-agnostic reference-relative physical target smoothness.

Standard action-rate penalty acts on normalized actions. G1 action scales span
0.0745–0.5475 rad, so equal angular changes receive very unequal penalties.
This experimental term penalizes unexpected target changes in physical units,
while retaining a small conventional action-rate component.
"""
import torch

def reference_relative_target_rate(env):
 action=env.action_manager.action
 delta=action-env.action_manager.prev_action
 scale=env.action_manager.get_term('joint_pos').scale
 motion=env.command_manager.get_term('motion')
 dt=env.step_dt
 expected=motion.joint_vel*dt
 residual=(delta*scale-expected)/.35
 # Do not charge reference velocity against a reset action at episode entry.
 mask=(env.episode_length_buf>1).to(action.dtype)
 return mask*(.875*residual.square().sum(-1)+.125*delta.square().sum(-1))

def pelvis_relative_distal_position(env):
 from mjlab.utils.lab_api.math import matrix_from_quat
 t=env.command_manager.get_term('motion');root=t.cfg.body_names.index('pelvis')
 ids=[t.cfg.body_names.index(n) for n in ('left_ankle_roll_link','right_ankle_roll_link','left_wrist_yaw_link','right_wrist_yaw_link')]
 target=t.body_pos_w[:,ids]-t.body_pos_w[:,root:root+1]
 actual=t.robot_body_pos_w[:,ids]-t.robot_body_pos_w[:,root:root+1]
 target=torch.einsum('bij,bkj->bki',matrix_from_quat(t.body_quat_w[:,root]).transpose(-1,-2),target)
 actual=torch.einsum('bij,bkj->bki',matrix_from_quat(t.robot_body_quat_w[:,root]).transpose(-1,-2),actual)
 return torch.exp(-((target-actual).norm(dim=-1)/.08).square()).mean(-1)
