"""Online residual RL around a frozen official SONIC controller.

PPO optimizes only its sampled residual. There is no action-label dataset or
teacher loss. Histories contain actions actually executed after residual addition
and clipping. Done environments reset their histories independently.
"""

import json
from pathlib import Path

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.utils.lab_api.math import matrix_from_quat
from mjlab.utils.lab_api.math import quat_apply_inverse
from mjlab.utils.lab_api.math import quat_inv
from mjlab.utils.lab_api.math import quat_mul
from tensordict import TensorDict
import torch

from .sonic import SonicModeZeroPolicy


class SonicResidualWrapper(RslRlVecEnvWrapper):
    def __init__(self, environment, directory, contract_path, clip_actions=None):
        super().__init__(environment, clip_actions=clip_actions)
        contract = json.loads(Path(contract_path).read_text())
        self.base = SonicModeZeroPolicy(directory, contract).eval().to(self.device)
        self.robot = self.unwrapped.scene["robot"]
        self.command = self.unwrapped.command_manager.get_term("motion")
        action = self.unwrapped.action_manager.get_term("joint_pos")
        if list(self.robot.joint_names) != contract["joint_names"]:
            raise ValueError("SONIC residual joint ABI differs from native contract")
        if list(action.target_names) != contract["action_target_names"]:
            raise ValueError("SONIC residual motor ABI differs from native contract")
        self.scale = action.scale
        self.offset = action.offset
        torch.testing.assert_close(self.scale[0], self.base.scale)
        torch.testing.assert_close(self.offset[0], self.base.offset)
        self.source_order = torch.tensor(
            [
                contract["joint_names"].index(name)
                for name in contract["sonic_source_joint_names"]
            ],
            device=self.device,
        )
        self.mujoco_to_isaac = torch.argsort(self.base.il_to_mj)
        self.inverse_action_order = torch.argsort(self.base.action_order)
        self.histories = torch.zeros(self.num_envs, 10, 93, device=self.device)
        self.initialized = torch.zeros(
            self.num_envs, dtype=torch.bool, device=self.device
        )
        self.previous_actions = torch.zeros(self.num_envs, 29, device=self.device)
        self.cached_observations = None

    @torch.no_grad()
    def refresh(self, observations):
        indices = torch.minimum(
            self.command.time_steps[:, None]
            + torch.arange(10, device=self.device)[None] * 5,
            self.command.ends[self.command.clip_ids, None] - 1,
        )
        encoder = torch.zeros(self.num_envs, 1762, device=self.device)
        reference_positions = self.command.motion.joint_pos[indices]
        reference_velocities = self.command.motion.joint_vel[indices]
        encoder[:, 4:294] = reference_positions[:, :, self.source_order][
            :, :, self.mujoco_to_isaac
        ].flatten(1)
        encoder[:, 294:584] = reference_velocities[:, :, self.source_order][
            :, :, self.mujoco_to_isaac
        ].flatten(1)
        root_quaternion = self.robot.data.root_link_quat_w
        reference_quaternions = self.command.motion._body_quat_w[indices, 0]
        relative = quat_mul(
            quat_inv(root_quaternion)[:, None].expand(-1, 10, -1), reference_quaternions
        )
        encoder[:, 601:661] = matrix_from_quat(relative)[:, :, :, :2].flatten(1)
        previous_targets = self.previous_actions * self.scale + self.offset
        previous_raw = (
            (previous_targets[:, self.inverse_action_order] - self.base.q0)
            / self.base.sonic_scale
        )[:, self.mujoco_to_isaac]
        previous_raw[~self.initialized] = 0
        gravity = root_quaternion.new_tensor([0.0, 0.0, -1.0]).expand(self.num_envs, -1)
        state = torch.cat(
            [
                self.robot.data.root_link_ang_vel_b,
                (self.robot.data.joint_pos[:, self.source_order] - self.base.q0)[
                    :, self.mujoco_to_isaac
                ],
                self.robot.data.joint_vel[:, self.source_order][
                    :, self.mujoco_to_isaac
                ],
                previous_raw,
                quat_apply_inverse(root_quaternion, gravity),
            ],
            dim=1,
        )
        shifted = torch.cat([self.histories[:, 1:], state[:, None]], dim=1)
        self.histories.copy_(
            torch.where(
                self.initialized[:, None, None],
                shifted,
                state[:, None].expand(-1, 10, -1),
            )
        )
        self.initialized.fill_(True)
        token = self.base.encoder(encoder)
        history = self.histories
        decoder_input = torch.cat(
            [
                token,
                history[:, :, :3].flatten(1),
                history[:, :, 3:32].flatten(1),
                history[:, :, 32:61].flatten(1),
                history[:, :, 61:90].flatten(1),
                history[:, :, 90:93].flatten(1),
            ],
            dim=1,
        )
        raw = self.base.decoder(decoder_input).clamp(-20, 20)
        target = self.base.q0 + raw[:, self.base.il_to_mj] * self.base.sonic_scale
        self.nominal_actions = (
            target[:, self.base.action_order] - self.offset
        ) / self.scale
        self.encoder_inputs = encoder
        self.sonic_states = state
        self.tokens = token
        self.cached_observations = TensorDict(
            {
                name: torch.cat([values, self.nominal_actions, token], dim=1)
                for name, values in observations.items()
            },
            batch_size=[self.num_envs],
        )
        return self.cached_observations

    def get_observations(self):
        if self.cached_observations is None:
            return self.refresh(super().get_observations())
        return self.cached_observations

    def reset(self):
        observations, extras = super().reset()
        self.initialized.zero_()
        self.previous_actions.zero_()
        return self.refresh(observations), extras

    def to_environment_actions(self, residual_actions):
        self.get_observations()
        actions = self.nominal_actions + residual_actions
        if self.clip_actions is not None:
            actions = actions.clamp(-self.clip_actions, self.clip_actions)
        return actions

    def step(self, actions):
        executed = self.to_environment_actions(actions)
        observations, rewards, dones, extras = super().step(executed)
        self.previous_actions.copy_(executed)
        self.initialized[dones.bool()] = False
        self.previous_actions[dones.bool()] = 0
        return self.refresh(observations), rewards, dones, extras
