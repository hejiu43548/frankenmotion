"""Reference-centered actions, with unchanged physical action observations/rewards.

The nominal command is kinematic joint position, never an expert policy output.
PPO's stochastic action remains the residual; the state-dependent translation is
applied only at the environment boundary. The environment therefore penalizes
changes in actual motor targets, exactly as in the absolute-action baseline.
"""

import torch
from mjlab.rl import RslRlVecEnvWrapper


class ReferenceResidualWrapper(RslRlVecEnvWrapper):
    def __init__(self, environment, clip_actions=None):
        super().__init__(environment, clip_actions=clip_actions)
        action = self.unwrapped.action_manager.get_term("joint_pos")
        if list(action.target_names) != list(self.unwrapped.scene["robot"].joint_names):
            raise ValueError("Reference and action joint ordering must match")
        self.scale = action.scale
        self.offset = action.offset

    def to_environment_actions(self, residual_actions):
        reference = self.unwrapped.command_manager.get_term("motion").joint_pos
        nominal_actions = (reference - self.offset) / self.scale
        return residual_actions + nominal_actions

    def step(self, actions):
        return super().step(self.to_environment_actions(actions))


class ReferenceResidualPolicy(torch.nn.Module):
    """Export emits ordinary normalized joint targets, compatible with NativeTracker."""

    def __init__(self, policy, action_scale, action_offset):
        super().__init__()
        self.policy = policy
        self.register_buffer("action_scale", torch.as_tensor(action_scale).float())
        self.register_buffer("action_offset", torch.as_tensor(action_offset).float())

    def forward(self, observations):
        residual = self.policy(observations)
        return (
            residual + (observations[:, :29] - self.action_offset) / self.action_scale
        )
