import numpy as np
import torch
from src.tools.geometry import (
    rotation_6d_to_matrix,
    matrix_to_euler_angles,
    axis_angle_rotation,
    matrix_to_axis_angle,
)
from .schema import TASKS


class FK:
    def __init__(self, skeleton, device="cpu"):
        skeleton_data = np.load(skeleton)
        self.rest_joint_positions = torch.tensor(
            skeleton_data["J"], device=device, dtype=torch.float32
        )
        self.parents = skeleton_data["parents"][:22].tolist()
        self.height = float(skeleton_data["height"])

    def __call__(self, motion, canonical=True, return_pose=False):
        # Same SMPL-RIFKE root reconstruction as upstream; no redundant xyz prediction.
        batch_size, num_frames, _ = motion.shape
        local_rotations = rotation_6d_to_matrix(
            motion[..., 4:136].reshape(batch_size, num_frames, 22, 6)
        )
        root_euler_angles = matrix_to_euler_angles(local_rotations[:, :, 0], "ZYX")
        yaw = torch.cat(
            [torch.zeros_like(motion[:, :1, 3]), torch.cumsum(motion[:, :-1, 3], 1)], 1
        )
        yaw_rotation = axis_angle_rotation("Z", yaw)
        root_rotation = (
            yaw_rotation
            @ axis_angle_rotation("Y", root_euler_angles[..., 1])
            @ axis_angle_rotation("X", root_euler_angles[..., 2])
        )
        reconstructed_rotations = torch.cat(
            [root_rotation[:, :, None], local_rotations[:, :, 1:]], 2
        )
        root_velocity = (yaw_rotation[..., :2, :2] @ motion[..., 1:3, None]).squeeze(-1)
        root_xy = torch.cat(
            [
                torch.zeros_like(root_velocity[:, :1]),
                torch.cumsum(root_velocity[:, :-1], 1),
            ],
            1,
        )
        root_translation = torch.cat([root_xy, motion[..., :1]], -1)
        global_rotations = []
        joint_positions = []
        for joint_index in range(22):
            if joint_index == 0:
                global_rotations.append(reconstructed_rotations[:, :, 0])
                joint_positions.append(
                    self.rest_joint_positions[0].expand(batch_size, num_frames, 3)
                    + root_translation
                )
            else:
                parent_index = self.parents[joint_index]
                global_rotations.append(
                    global_rotations[parent_index]
                    @ reconstructed_rotations[:, :, joint_index]
                )
                joint_positions.append(
                    joint_positions[parent_index]
                    + (
                        global_rotations[parent_index]
                        @ (
                            self.rest_joint_positions[joint_index]
                            - self.rest_joint_positions[parent_index]
                        )[:, None]
                    ).squeeze(-1)
                )
        joint_positions.extend(
            [
                joint_positions[20]
                + (
                    global_rotations[20]
                    @ (self.rest_joint_positions[22] - self.rest_joint_positions[20])[
                        :, None
                    ]
                ).squeeze(-1),
                joint_positions[21]
                + (
                    global_rotations[21]
                    @ (self.rest_joint_positions[37] - self.rest_joint_positions[21])[
                        :, None
                    ]
                ).squeeze(-1),
            ]
        )
        joint_positions = torch.stack(joint_positions, 2)
        if canonical:
            side = joint_positions[:, 0, 1, :2] - joint_positions[:, 0, 2, :2]
            angle = torch.atan2(side[:, 1], side[:, 0]) - np.pi / 2
            canonical_rotation = axis_angle_rotation("Z", -angle)
            joint_positions = (
                canonical_rotation[:, None, None] @ joint_positions[..., None]
            ).squeeze(-1)
        if return_pose:
            return (
                joint_positions,
                matrix_to_axis_angle(reconstructed_rotations).reshape(
                    batch_size, num_frames, 66
                ),
                root_translation,
            )
        return joint_positions


def quantity(joint_positions, task, scale=1.0, net_walk=False):
    # Frozen screenshot quantities, including legacy path-speed walk; net-speed is separately reported.
    task = TASKS[task] if isinstance(task, int) else task
    root = joint_positions[:, :, 0]
    duration_seconds = (joint_positions.shape[1] - 1) / 20
    if task == "raise_hand":
        measured_quantity = torch.quantile(
            (joint_positions[:, :, 21] - root)[..., 2], 0.95, dim=1
        )
    elif task == "reach":
        measured_quantity = torch.quantile(
            (joint_positions[:, :, 21] - root)[..., 0], 0.95, dim=1
        )
    elif task == "strike":
        wrist_positions = joint_positions[:, :, 21]
        smoothed_wrist_positions = (
            0.25 * wrist_positions[:, :-2]
            + 0.5 * wrist_positions[:, 1:-1]
            + 0.25 * wrist_positions[:, 2:]
        )
        wrist_speed = (
            smoothed_wrist_positions[:, 2:] - smoothed_wrist_positions[:, :-2]
        ).norm(dim=-1) * 10
        measured_quantity = wrist_speed[:, 14:32].amax(1)
    elif task == "wave":
        wave_positions = joint_positions[:, 16:101]
        lateral_direction = wave_positions[:, :, 16] - wave_positions[:, :, 17]
        lateral_direction = lateral_direction / lateral_direction.norm(
            dim=-1, keepdim=True
        ).clamp_min(1e-8)
        lateral_wrist_displacement = (
            (
                wave_positions[:, :, 21]
                - (wave_positions[:, :, 16] + wave_positions[:, :, 17]) / 2
            )
            * lateral_direction
        ).sum(-1)
        measured_quantity = (
            torch.quantile(lateral_wrist_displacement, 0.95, dim=1)
            - torch.quantile(lateral_wrist_displacement, 0.05, dim=1)
        ) / 2
    elif task == "turn":
        side = joint_positions[:, :, 1, :2] - joint_positions[:, :, 2, :2]
        yaw = torch.atan2(side[..., 1], side[..., 0])
        yaw_difference = yaw[:, -1] - yaw[:, 0]
        return -torch.atan2(torch.sin(yaw_difference), torch.cos(yaw_difference))
    elif task == "sidestep":
        measured_quantity = -(root[:, :, 1] - root[:, :1, 1]).amin(1)
    elif task == "back_walk":
        measured_quantity = -(root[:, -1, 0] - root[:, 0, 0]) / duration_seconds
    elif task == "kick":
        ankle_forward_displacement = (joint_positions[:, :, 8] - root)[..., 0]
        measured_quantity = (
            ankle_forward_displacement[:, 10:51].amax(1)
            - ankle_forward_displacement[:, 0]
        )
    elif task == "jump":
        measured_quantity = root[:, :, 2].amax(1) - root[:, 0, 2]
    elif task == "lean":
        torso_vector = (
            joint_positions[:, :, 16] + joint_positions[:, :, 17]
        ) / 2 - root
        pitch = torch.atan2(torso_vector[..., 0], torso_vector[..., 2])
        return torch.quantile(pitch[:, 20:59], 0.9, dim=1)
    elif task == "walk":
        measured_quantity = (
            (root[:, -1, 0] - root[:, 0, 0]) / duration_seconds
            if net_walk
            else (root[:, 1:, :2] - root[:, :-1, :2]).norm(dim=-1).sum(1)
            / duration_seconds
        )
    else:
        raise ValueError(task)
    return measured_quantity * scale
