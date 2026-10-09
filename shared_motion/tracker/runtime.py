"""One stateful exported SONIC backbone + one shared residual head."""

import torch


class Tracker:
    def __init__(self, checkpoint, device="cpu"):
        self.device = device
        self.actor = torch.jit.load(str(checkpoint), map_location=device).eval()
        self.reset()

    def reset(self):
        self.actor.reset()

    @torch.inference_mode()
    def __call__(self, observation):
        observation_tensor = torch.as_tensor(
            observation, dtype=torch.float32, device=self.device
        )
        if observation_tensor.shape != (1, 2350):
            raise ValueError("Expected [1,2350]; history belongs to one environment")
        action = self.actor(observation_tensor)
        if action.shape != (1, 29) or not torch.isfinite(action).all():
            raise RuntimeError("Invalid tracker action")
        return action
