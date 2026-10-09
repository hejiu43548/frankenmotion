"""FK matching the retrieved Charlie experiment's pelvis-height convention."""

import numpy as np
import torch
from torch import nn
from src.tools.geometry import (
    rotation_6d_to_matrix,
    matrix_to_euler_angles,
    axis_angle_rotation,
)


class Skeleton(nn.Module):
    def __init__(self, path):
        super().__init__()
        archive = np.load(path)
        self.register_buffer(
            "rest_positions", torch.tensor(archive["J"], dtype=torch.float32)
        )
        self.parents = archive["parents"][:22].tolist()
        self.height = float(archive["height"])

    def forward(self, motion):
        batch_size, frames, _ = motion.shape
        rotations = rotation_6d_to_matrix(
            motion[..., 4:136].reshape(batch_size, frames, 22, 6)
        )
        angles = matrix_to_euler_angles(rotations[:, :, 0], "ZYX")
        yaw = torch.cat(
            [torch.zeros_like(motion[:, :1, 3]), motion[:, :-1, 3].cumsum(1)], 1
        )
        yaw_rotation = axis_angle_rotation("Z", yaw)
        root_rotation = (
            yaw_rotation
            @ axis_angle_rotation("Y", angles[..., 1])
            @ axis_angle_rotation("X", angles[..., 2])
        )
        rotations = torch.cat([root_rotation[:, :, None], rotations[:, :, 1:]], 2)
        velocity = (yaw_rotation[..., :2, :2] @ motion[..., 1:3, None]).squeeze(-1)
        root_xy = torch.cat(
            [torch.zeros_like(velocity[:, :1]), velocity[:, :-1].cumsum(1)], 1
        )
        positions = [torch.cat([root_xy, motion[..., :1]], -1)]
        global_rotations = [rotations[:, :, 0]]
        for joint_index in range(1, 22):
            parent = self.parents[joint_index]
            global_rotations.append(
                global_rotations[parent] @ rotations[:, :, joint_index]
            )
            offset = self.rest_positions[joint_index] - self.rest_positions[parent]
            positions.append(
                positions[parent]
                + (global_rotations[parent] @ offset[:, None]).squeeze(-1)
            )
        for joint_index, parent in [(22, 20), (37, 21)]:
            offset = self.rest_positions[joint_index] - self.rest_positions[parent]
            positions.append(
                positions[parent]
                + (global_rotations[parent] @ offset[:, None]).squeeze(-1)
            )
        positions = torch.stack(positions, -2)
        side = positions[:, 0, 1, :2] - positions[:, 0, 2, :2]
        heading = torch.atan2(side[:, 1], side[:, 0]) - np.pi / 2
        return (
            axis_angle_rotation("Z", -heading)[:, None, None] @ positions[..., None]
        ).squeeze(-1)
