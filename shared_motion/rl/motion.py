"""Task-balanced reference sampling with explicit finite episode boundaries."""

from dataclasses import dataclass
import json
from pathlib import Path

import torch
from mjlab.tasks.tracking.mdp.commands import MotionCommand
from mjlab.tasks.tracking.mdp.commands import MotionCommandCfg
from mjlab.utils.lab_api.math import matrix_from_quat
from mjlab.utils.lab_api.math import quat_apply
from mjlab.utils.lab_api.math import yaw_quat
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.lab_api.math import quat_inv
from mjlab.utils.lab_api.math import quat_mul


class MultiClipCommand(MotionCommand):
    def __init__(self, configuration, environment):
        super().__init__(configuration, environment)
        metadata = json.loads(Path(configuration.metadata).read_text())
        self.ends = torch.tensor(metadata["ends"], device=self.device, dtype=torch.long)
        self.starts = torch.cat([self.ends.new_zeros(1), self.ends[:-1]])
        if self.ends[-1] != self.motion.time_step_total or torch.any(
            self.ends - self.starts < 3
        ):
            raise ValueError("Motion archive and clip boundaries disagree")
        self.clip_ids = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.sample_counts = torch.zeros_like(self.ends)
        tasks = sorted({record["task"] for record in metadata["records"]})
        groups = [
            [
                index
                for index, record in enumerate(metadata["records"])
                if record["task"] == task
            ]
            for task in tasks
        ]
        self.task_sizes = torch.tensor(
            [len(group) for group in groups], device=self.device
        )
        self.task_clips = torch.zeros(
            len(groups), max(map(len, groups)), device=self.device, dtype=torch.long
        )
        for task_index, group in enumerate(groups):
            self.task_clips[task_index, : len(group)] = torch.tensor(
                group, device=self.device
            )
        self.environment_tasks = torch.arange(self.num_envs, device=self.device) % len(
            groups
        )

    def _uniform_sampling(self, environment_ids):
        if self.cfg.evaluation:
            clip_ids = environment_ids % len(self.ends)
            offsets = torch.zeros_like(clip_ids)
        else:
            task_ids = self.environment_tasks[environment_ids]
            choices = (
                torch.rand(len(environment_ids), device=self.device)
                * self.task_sizes[task_ids]
            ).long()
            clip_ids = self.task_clips[task_ids, choices]
            lengths = self.ends[clip_ids] - self.starts[clip_ids]
            offsets = (
                torch.rand(len(clip_ids), device=self.device) * (lengths - 2)
            ).long()
            offsets[
                torch.rand(len(clip_ids), device=self.device)
                < self.cfg.start_probability
            ] = 0
        self.clip_ids[environment_ids] = clip_ids
        self.time_steps[environment_ids] = self.starts[clip_ids] + offsets
        self.sample_counts += torch.bincount(clip_ids, minlength=len(self.ends))

    def refresh_reference_frame(self):
        """Refresh relative targets after FK, without advancing reference time.

        Mirrors the upstream MotionCommand target-frame transform. Explicit
        environment reset computes FK but does not call command.compute().
        """
        anchor_position = self.anchor_pos_w[:, None, :]
        anchor_orientation = self.anchor_quat_w[:, None, :]
        robot_position = self.robot_anchor_pos_w[:, None, :].clone()
        robot_position[..., 2] = anchor_position[..., 2]
        relative_yaw = yaw_quat(
            quat_mul(self.robot_anchor_quat_w[:, None, :], quat_inv(anchor_orientation))
        )
        relative_yaw = relative_yaw.expand(-1, len(self.cfg.body_names), -1)
        self.body_quat_relative_w = quat_mul(relative_yaw, self.body_quat_w)
        self.body_pos_relative_w = robot_position + quat_apply(
            relative_yaw, self.body_pos_w - anchor_position
        )

    def _update_command(self):
        if torch.any(self.time_steps + 1 >= self.ends[self.clip_ids]):
            raise RuntimeError(
                "Reference boundary crossed without an episode termination"
            )
        super()._update_command()


@dataclass(kw_only=True)
class MultiClipCommandCfg(MotionCommandCfg):
    metadata: str
    start_probability: float = 0.35
    evaluation: bool = False

    def build(self, environment):
        return MultiClipCommand(self, environment)


def reference_finished(environment):
    command = environment.command_manager.get_term("motion")
    return command.time_steps >= command.ends[command.clip_ids] - 1


def reference_preview(environment, offsets):
    command = environment.command_manager.get_term("motion")
    inverse_orientation = quat_inv(command.robot_anchor_quat_w)
    anchor_index = command.body_indexes[command.motion_anchor_body_index]
    observations = []
    for offset in offsets:
        indices = torch.minimum(
            command.time_steps + offset, command.ends[command.clip_ids] - 1
        )
        anchor_position = (
            command.motion._body_pos_w[indices, anchor_index]
            + environment.scene.env_origins
        )
        anchor_orientation = command.motion._body_quat_w[indices, anchor_index]
        position = quat_apply_inverse(
            command.robot_anchor_quat_w, anchor_position - command.robot_anchor_pos_w
        )
        orientation = matrix_from_quat(
            quat_mul(inverse_orientation, anchor_orientation)
        )[:, :, :2].reshape(-1, 6)
        observations.extend(
            [
                command.motion.joint_pos[indices],
                command.motion.joint_vel[indices],
                position,
                orientation,
            ]
        )
    return torch.cat(observations, dim=-1)
