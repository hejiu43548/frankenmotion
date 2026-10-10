"""Layer-residual adapters with stable task IDs and three-component commands.

RootControl retains the imported scalar-root protocol. TaskControl accepts XYZ
reach targets and masked scalar commands for the other eighteen active tasks.
The retired point embedding slot is reserved; old task adapters are incompatible.
"""

import torch
from torch import nn


class RootControl(nn.Module):
    def __init__(self, base, bottleneck=128):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        width = base.latent_dim
        self.encoder = nn.Sequential(
            nn.Linear(4, bottleneck), nn.SiLU(), nn.Linear(bottleneck, width)
        )
        self.residuals = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(width, bottleneck),
                    nn.SiLU(),
                    nn.Linear(bottleneck, width),
                )
                for _ in base.seqTransEncoder.layers
            ]
        )
        for residual in self.residuals:
            nn.init.zeros_(residual[-1].weight)
            nn.init.zeros_(residual[-1].bias)
        self.condition = None
        self.handles = [
            layer.register_forward_hook(self.make_hook(layer_index))
            for layer_index, layer in enumerate(base.seqTransEncoder.layers)
        ]

    def make_hook(self, layer_index):
        def apply_residual(module, arguments, output):
            if self.condition is None:
                return output
            features, valid = self.condition
            residual = self.residuals[layer_index](features) * valid
            return output + torch.nn.functional.pad(
                residual, (0, 0, output.shape[1] - residual.shape[1], 0)
            )

        return apply_residual

    def forward(self, motion, conditioning, timesteps):
        control = conditioning.get("root_control")
        self.condition = (
            None
            if control is None
            else (self.encoder(control), control[..., 2:].amax(-1, keepdim=True))
        )
        try:
            return self.base(motion, conditioning, timesteps)
        finally:
            self.condition = None

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self


class TaskControl(nn.Module):
    def __init__(self, root, command_ranges, bottleneck=128, embedding_width=32):
        super().__init__()
        self.root = root
        self.root.requires_grad_(False)
        width = root.base.latent_dim
        self.task = nn.Embedding(len(command_ranges), embedding_width)
        self.encoder = nn.Sequential(
            nn.Linear(embedding_width + 3, bottleneck),
            nn.SiLU(),
            nn.Linear(bottleneck, width),
        )
        self.residuals = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(width, bottleneck),
                    nn.SiLU(),
                    nn.Linear(bottleneck, width),
                )
                for _ in root.base.seqTransEncoder.layers
            ]
        )
        self.command_ranges = command_ranges
        for residual in self.residuals:
            nn.init.zeros_(residual[-1].weight)
            nn.init.zeros_(residual[-1].bias)
        self.condition = None
        self.handles = [
            layer.register_forward_hook(self.make_hook(layer_index))
            for layer_index, layer in enumerate(root.base.seqTransEncoder.layers)
        ]

    def encoded(self, task_indices, commands):
        from .reach import normalized_commands

        normalized = normalized_commands(commands, task_indices, self.command_ranges)
        return self.encoder(torch.cat([self.task(task_indices), normalized], -1))

    def outputs(self, task_indices, commands):
        features = self.encoded(task_indices, commands)
        return torch.stack([residual(features) for residual in self.residuals], -2)

    def make_hook(self, layer_index):
        def apply_residual(module, arguments, output):
            if self.condition is None:
                return output
            features, frames = self.condition
            residual = self.residuals[layer_index](features)[:, None].expand(
                -1, frames, -1
            )
            return output + torch.nn.functional.pad(
                residual, (0, 0, output.shape[1] - frames, 0)
            )

        return apply_residual

    def forward(self, motion, conditioning, timesteps):
        self.condition = None
        if "task_control" in conditioning:
            task_indices, commands = conditioning["task_control"]
            self.condition = (self.encoded(task_indices, commands), motion.shape[1])
        try:
            return self.root(motion, conditioning, timesteps)
        finally:
            self.condition = None

    def train(self, mode=True):
        super().train(mode)
        self.root.eval()
        return self
