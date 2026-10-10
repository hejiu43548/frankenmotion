"""Bounded arm-rotation residual policy trained by one-step PPO feedback.

A contextual-bandit policy observes this experiment's generated motion and the
requested XYZ. Its nine actions rotate right collar/shoulder/elbow. A fixed smooth
window and bounded rotations preserve the rest of the generated motion. There is
no inverse-kinematics target, teacher action or action imitation loss.
"""

import math

import torch
from torch import nn

from src.tools.geometry import (
    axis_angle_to_matrix,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
    matrix_to_euler_angles,
    axis_angle_rotation,
)
from .reach3d_geometry import wrist_positions_in_body_frame


CONTROL_JOINTS = [14, 17, 19]


class StrikeResidualPolicy(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.mean = nn.Sequential(
            nn.Linear(27, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 9),
        )
        nn.init.zeros_(self.mean[-1].weight)
        nn.init.zeros_(self.mean[-1].bias)
        self.log_std = nn.Parameter(torch.full((9,), -1.5))

    def distribution(self, observations):
        return torch.distributions.Normal(
            self.mean(observations), self.log_std.clamp(-3.0, -0.3).exp()
        )


def observations(motion, target, event_frames, skeleton):
    joints = skeleton(motion)
    wrists = wrist_positions_in_body_frame(joints[..., :22, :])
    batch_indices = torch.arange(len(motion), device=motion.device)
    wrist = wrists[batch_indices, event_frames, 1]
    rotations = motion[..., 4:136].reshape(*motion.shape[:2], 22, 6)
    selected = rotations[batch_indices, event_frames][:, CONTROL_JOINTS].flatten(1)
    return torch.cat([selected, wrist, target, (target - wrist) / 0.2], dim=-1)


def apply_residual(
    motion,
    raw_action,
    event_frames,
    skeleton,
    max_angle=0.35,
    radius_frames=12,
    hands=None,
):
    if (
        raw_action.shape != (len(motion), 9)
        or radius_frames <= 0
        or not 0 < max_angle <= 0.7
    ):
        raise ValueError("Invalid bounded arm residual arguments")
    rotations = rotation_6d_to_matrix(
        motion[..., 4:136].reshape(*motion.shape[:2], 22, 6)
    )
    phase = (
        torch.arange(motion.shape[1], device=motion.device)[None]
        - event_frames[:, None]
    ) / radius_frames
    envelope = 0.5 * (1 + torch.cos(math.pi * phase.clamp(-1, 1)))
    delta = raw_action.tanh().reshape(len(motion), 3, 3) * max_angle
    corrections = axis_angle_to_matrix(delta[:, None] * envelope[:, :, None, None])
    corrected = motion.clone()
    if hands is None:
        hands = torch.ones(len(motion), device=motion.device, dtype=torch.long)
    if hands.shape != (len(motion),) or torch.any((hands < 0) | (hands > 1)):
        raise ValueError("hands must contain 0=left or 1=right for each sequence")
    corrected_features = corrected[..., 4:136].reshape(*motion.shape[:2], 22, 6)
    batch_indices = torch.arange(len(motion), device=motion.device)
    control = torch.tensor([[13, 16, 18], [14, 17, 19]], device=motion.device)[hands]
    for control_index in range(3):
        joint_indices = control[:, control_index]
        updated = (
            rotations[batch_indices, :, joint_indices]
            @ corrections[:, :, control_index]
        )
        corrected_features[batch_indices, :, joint_indices] = matrix_to_rotation_6d(
            updated
        )
    # Keep the redundant 23 joint-position channels consistent with corrected FK.
    joints = skeleton(corrected)
    relative = joints[:, :, 1:] - joints[:, :, :1]
    root_angles = matrix_to_euler_angles(rotations[:, 0, 0], "ZYX")
    root_orientation = axis_angle_rotation(
        "Y", root_angles[:, 1]
    ) @ axis_angle_rotation("X", root_angles[:, 2])
    initial_side = (
        root_orientation
        @ (skeleton.rest_positions[1] - skeleton.rest_positions[2])[:, None]
    ).squeeze(-1)
    initial_heading = torch.atan2(initial_side[:, 1], initial_side[:, 0]) - math.pi / 2
    yaw = torch.cat(
        [torch.zeros_like(motion[:, :1, 3]), motion[:, :-1, 3].cumsum(1)], 1
    )
    local_rotation = axis_angle_rotation("Z", initial_heading[:, None] - yaw)
    local_joints = (local_rotation[:, :, None] @ relative[..., None]).squeeze(-1)
    corrected[..., 136:] = local_joints.flatten(-2)
    return corrected
