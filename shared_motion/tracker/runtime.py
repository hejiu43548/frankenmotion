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
        x = torch.as_tensor(observation, dtype=torch.float32, device=self.device)
        if x.shape != (1, 2350):
            raise ValueError("Expected [1,2350]; history belongs to one environment")
        y = self.actor(x)
        if y.shape != (1, 29) or not torch.isfinite(y).all():
            raise RuntimeError("Invalid tracker action")
        return y
