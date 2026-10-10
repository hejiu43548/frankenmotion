"""New hand/XYZ-conditioned layer residuals over a frozen official denoiser.

No historical RootControl, specialist adapter, or teacher policy is loaded here.
This module is an architecture foundation; dataset ranges and training protocol
are supplied separately after inspecting the actual supervision.
"""

import torch
from torch import nn


class ReachTargetResidual(nn.Module):
    def __init__(
        self,
        base,
        target_center,
        target_scale,
        bottleneck=128,
        hand_embedding_width=16,
    ):
        super().__init__()
        center = torch.as_tensor(target_center, dtype=torch.float32)
        scale = torch.as_tensor(target_scale, dtype=torch.float32)
        if center.shape != (3,) or scale.shape != (3,):
            raise ValueError("Target normalization must have three coordinates")
        if not torch.isfinite(center).all() or not torch.isfinite(scale).all():
            raise ValueError("Target normalization must be finite")
        if torch.any(scale <= 0):
            raise ValueError("Every target normalization scale must be positive")
        self.register_buffer("target_center", center.clone())
        self.register_buffer("target_scale", scale.clone())
        self.base = base
        self.base.requires_grad_(False)
        self.base.eval()
        self.hand_embedding = nn.Embedding(2, hand_embedding_width)
        self.encoder = nn.Sequential(
            nn.Linear(hand_embedding_width + 3, bottleneck),
            nn.SiLU(),
            nn.Linear(bottleneck, base.latent_dim),
        )
        self.residuals = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(base.latent_dim, bottleneck),
                    nn.SiLU(),
                    nn.Linear(bottleneck, base.latent_dim),
                )
                for _ in base.seqTransEncoder.layers
            ]
        )
        for residual in self.residuals:
            nn.init.zeros_(residual[-1].weight)
            nn.init.zeros_(residual[-1].bias)
        self._condition = None
        self._handles = [
            layer.register_forward_hook(self._make_hook(layer_index))
            for layer_index, layer in enumerate(base.seqTransEncoder.layers)
        ]

    def encode_targets(self, hands, positions):
        if hands.ndim != 1 or positions.shape != (len(hands), 3):
            raise ValueError("Expected hands[B] and positions[B,3]")
        if hands.dtype != torch.long or torch.any((hands < 0) | (hands > 1)):
            raise ValueError("Hand identity must be int64: left=0, right=1")
        if not torch.isfinite(positions).all():
            raise ValueError("Target positions must be finite")
        normalized = (positions - self.target_center) / self.target_scale
        return self.encoder(torch.cat([self.hand_embedding(hands), normalized], -1))

    def _make_hook(self, layer_index):
        def add_residual(module, arguments, output):
            if self._condition is None:
                return output
            features, frame_mask = self._condition
            frame_count = frame_mask.shape[1]
            residual = self.residuals[layer_index](features)[:, None]
            residual = residual * frame_mask[..., None].to(residual)
            prefix = output.new_zeros(
                output.shape[0], output.shape[1] - frame_count, output.shape[2]
            )
            return output + torch.cat([prefix, residual], dim=1)

        return add_residual

    def forward(self, motion, conditioning, timesteps):
        target = conditioning.get("reach_target")
        self._condition = None
        if target is not None:
            features = self.encode_targets(target["hands"], target["positions"])
            if len(features) != len(motion):
                raise ValueError("Target and motion batches must match")
            if conditioning["mask"].shape != motion.shape[:2]:
                raise ValueError("Motion mask must match batch and frame dimensions")
            self._condition = (features, conditioning["mask"])
        try:
            return self.base(motion, conditioning, timesteps)
        finally:
            self._condition = None

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()
        return self

    def remove_hooks(self):
        """Release hooks before discarding a wrapper or reusing its backbone."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()
