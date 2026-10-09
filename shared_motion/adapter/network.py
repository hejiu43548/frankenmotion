"""Deployment control: ONE shared encoder, fixed dense heads, no legacy modules."""

import torch
from torch import nn
from .schema import availability

COMMAND_FEATURE_COUNT = 55


class SharedCommands(nn.Module):
    def __init__(self, width=512):
        super().__init__()
        self.width = width
        self.encoder = nn.Sequential(
            nn.Linear(COMMAND_FEATURE_COUNT, width),
            nn.SiLU(),
            nn.Linear(width, width),
            nn.SiLU(),
        )
        self.residuals = nn.ModuleList(
            [
                nn.Sequential(nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 512))
                for _ in range(4)
            ]
        )
        self.output = nn.Sequential(
            nn.Linear(width, width), nn.SiLU(), nn.Linear(width, 205)
        )

    def forward(self, command_features):
        encoded_commands = self.encoder(command_features)
        gate = availability(command_features)
        return (
            torch.stack(
                [residual_head(encoded_commands) for residual_head in self.residuals],
                -2,
            )
            * gate[..., None],
            self.output(encoded_commands) * gate,
        )


class UnifiedControl(nn.Module):
    def __init__(self, base, controller):
        super().__init__()
        self.base = base
        self.base.requires_grad_(False)
        self.controller = controller
        self.command_features = None
        self.static_residuals = None
        self.cached = None
        self.handles = [
            layer.register_forward_hook(self.hook(layer_index))
            for layer_index, layer in enumerate(base.seqTransEncoder.layers)
        ]

    def hook(self, layer_index):
        def add(module, args, layer_output):
            if self.cached is None:
                return layer_output
            layer_residual = self.cached[0][..., layer_index, :]
            return layer_output + torch.nn.functional.pad(
                layer_residual,
                (0, 0, layer_output.shape[1] - layer_residual.shape[1], 0),
            )

        return add

    def forward(self, motion, conditioning, timesteps, final_timesteps=None):
        self.cached = (
            self.static_residuals
            if self.static_residuals is not None
            else (
                None
                if self.command_features is None
                else self.controller(self.command_features)
            )
        )
        try:
            prediction = self.base(motion, conditioning, timesteps, final_timesteps)
            if self.cached is not None:
                prediction = prediction + torch.nn.functional.pad(
                    self.cached[1], (0, prediction.shape[-1] - 205)
                )
            return prediction
        finally:
            self.cached = None

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self
