"""Task-independent rewards for preserving global and distal reference motion.

No prompt, motion category, phase ID, or table coordinates enter these formulas.
Dense terms complement narrow exponential tracking rewards when errors are large.
"""
import torch
from mjlab.utils.lab_api.math import quat_error_magnitude


def _term(env):
    return env.command_manager.get_term('motion')


def pelvis_position_dense(env):
    t = _term(env)
    i = t.cfg.body_names.index('pelvis')
    err = (t.body_pos_w[:, i] - t.robot_body_pos_w[:, i]).norm(dim=-1)
    return -torch.nn.functional.huber_loss(err, torch.zeros_like(err), delta=.15, reduction='none') / .15


def pelvis_orientation(env):
    t = _term(env)
    i = t.cfg.body_names.index('pelvis')
    err = quat_error_magnitude(t.body_quat_w[:, i], t.robot_body_quat_w[:, i])
    return torch.exp(-(err / .3).square())


def distal_position(env):
    t = _term(env)
    ids = [t.cfg.body_names.index(n) for n in ('left_ankle_roll_link', 'right_ankle_roll_link', 'left_wrist_yaw_link', 'right_wrist_yaw_link')]
    err = (t.body_pos_w[:, ids] - t.robot_body_pos_w[:, ids]).norm(dim=-1)
    return torch.exp(-(err / .1).square()).mean(-1)


def reference_velocity_dense(env):
    t = _term(env)
    ids = [t.cfg.body_names.index(n) for n in ('pelvis', 'left_ankle_roll_link', 'right_ankle_roll_link', 'left_wrist_yaw_link', 'right_wrist_yaw_link')]
    err = (t.body_lin_vel_w[:, ids] - t.robot_body_lin_vel_w[:, ids]).norm(dim=-1)
    return -torch.nn.functional.huber_loss(err, torch.zeros_like(err), delta=1., reduction='none').mean(-1)


def bilateral_foot_height_dense(env):
    t = _term(env)
    ids = [t.cfg.body_names.index(n) for n in ('left_ankle_roll_link', 'right_ankle_roll_link')]
    err = t.body_pos_w[:, ids, 2] - t.robot_body_pos_w[:, ids, 2]
    return -torch.nn.functional.huber_loss(err, torch.zeros_like(err), delta=.08, reduction='none').mean(-1) / .08
