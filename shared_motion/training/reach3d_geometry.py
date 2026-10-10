"""Metric wrist coordinates in an instantaneous, yaw-aligned pelvis frame.

Axes are forward, left, up. Vertical remains gravity aligned, so torso lean does
not rotate the target vertically. This differs from a full torso rotation frame.
SMPL joints 1/2 are left/right hips; 20/21 are left/right wrists.
"""

import torch


def wrist_positions_in_body_frame(joint_positions):
    """Return (..., two hands, xyz), preserving input distance units and scale."""
    if joint_positions.ndim < 2 or joint_positions.shape[-2:] != (22, 3):
        raise ValueError("Expected (..., 22 SMPL joints, 3 coordinates)")
    if not torch.isfinite(joint_positions).all():
        raise ValueError("Joint coordinates must be finite")
    hip_difference = joint_positions[..., 1, :] - joint_positions[..., 2, :]
    horizontal_left = torch.cat(
        [hip_difference[..., :2], torch.zeros_like(hip_difference[..., 2:])], -1
    )
    hip_width = torch.linalg.vector_norm(horizontal_left, dim=-1, keepdim=True)
    if torch.any(hip_width < 1e-6):
        raise ValueError("Degenerate horizontal hip axis cannot define body yaw")
    left_axis = horizontal_left / hip_width
    up_axis = torch.zeros_like(left_axis)
    up_axis[..., 2] = 1
    forward_axis = torch.linalg.cross(left_axis, up_axis, dim=-1)
    world_from_body = torch.stack([forward_axis, left_axis, up_axis], -1)
    relative_wrists = joint_positions[..., [20, 21], :] - joint_positions[..., 0:1, :]
    return relative_wrists @ world_from_body
