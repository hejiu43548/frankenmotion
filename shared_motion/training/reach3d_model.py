"""Real-motion denoising supervision and deterministic reach-conditioned DDIM.

The caller supplies official-backbone normalizers and audited real-motion data.
Sampling has no motion/reference argument: only text, duration, hand and XYZ.
No learned root controller, teacher policy or teacher trajectory is used.
"""

import torch
from torch import nn

from .reach3d_adapter import ReachTargetResidual
from .reach3d_geometry import wrist_positions_in_body_frame


class ReachDiffusion(nn.Module):
    def __init__(self, bundle, target_center, target_scale, bottleneck=128):
        super().__init__()
        self.backbone_sha256 = bundle["sha256"]
        self.denoiser = ReachTargetResidual(
            bundle["denoiser"], target_center, target_scale, bottleneck=bottleneck
        )
        for name in ["mean", "std", "text_mean", "text_std", "alphas"]:
            self.register_buffer(name, bundle[name].detach().clone())
        if self.mean.numel() != 613 or self.std.numel() != 613:
            raise ValueError("Official motion/text features must have width 613")
        if self.alphas.ndim != 1 or not 2 <= len(self.alphas):
            raise ValueError("Expected at least two cumulative diffusion alphas")
        if not torch.all((self.alphas > 0) & (self.alphas < 1)):
            raise ValueError("Diffusion alphas must be strictly between zero and one")
        if not torch.all(self.alphas[1:] < self.alphas[:-1]):
            raise ValueError("Diffusion alphas must decrease with timestep")

    def conditioning(
        self, text, local, local_mask, mask, hands, positions, enabled=True
    ):
        if mask.ndim != 2 or mask.dtype != torch.bool:
            raise ValueError("Expected boolean motion mask [batch, frames]")
        batch_size, frames = mask.shape
        if not mask.any(dim=1).all():
            raise ValueError("Every sequence must contain a valid frame")
        if torch.any(mask[:, 1:] & ~mask[:, :-1]):
            raise ValueError("Motion masks must contain contiguous valid prefixes")
        if text.shape != (batch_size, 512):
            raise ValueError("Expected sequence text embeddings [batch, 512]")
        if local.shape != (batch_size, frames, 408) or local_mask.shape != local.shape:
            raise ValueError("Expected local text and mask [batch, frames, 408]")
        if local_mask.dtype != torch.bool:
            raise ValueError("Local text availability must be boolean")
        if not torch.isfinite(text).all() or not torch.isfinite(local).all():
            raise ValueError("Text embeddings must be finite")
        normalized_local = (local - self.mean[205:]) / (self.std[205:] + 1e-12)
        normalized_local = normalized_local * local_mask * mask[..., None]
        normalized_text = (text - self.text_mean) / (self.text_std + 1e-12)
        conditioning = {
            "mask": mask,
            "tx": {
                "x": normalized_text[:, None],
                "mask": torch.ones(batch_size, 1, device=text.device, dtype=torch.bool),
            },
        }
        if enabled:
            conditioning["reach_target"] = {"hands": hands, "positions": positions}
        return normalized_local, conditioning

    def decode_motion(self, normalized_motion):
        return normalized_motion * (self.std[:205] + 1e-12) + self.mean[:205]

    def supervised_loss(
        self,
        batch,
        generator,
        skeleton=None,
        endpoint_weight=0.0,
        target_frame="instantaneous_body",
    ):
        """Uniform-timestep x0 reconstruction of real observations.

        Optional endpoint supervision uses the reviewed event frame shared by all
        three coordinates. It must never use separate per-axis extrema or an
        endpoint frame selected from the prediction to make errors look smaller.
        """
        if endpoint_weight < 0:
            raise ValueError("Endpoint loss weight must be nonnegative")
        local, conditioning = self.conditioning(
            batch["tx"],
            batch["local"],
            batch["local_mask"],
            batch["mask"],
            batch["hands"],
            batch["positions"],
        )
        motion = batch["motion"]
        if (
            motion.shape != (*batch["mask"].shape, 205)
            or not torch.isfinite(motion).all()
        ):
            raise ValueError("Expected finite real motion [batch, frames, 205]")
        clean = (motion - self.mean[:205]) / (self.std[:205] + 1e-12)
        timesteps = torch.randint(
            len(self.alphas), (len(motion),), device=motion.device, generator=generator
        )
        noise = torch.randn(
            motion.shape, device=motion.device, dtype=motion.dtype, generator=generator
        )
        alpha = self.alphas[timesteps, None, None]
        noisy = (alpha.sqrt() * clean + (1 - alpha).sqrt() * noise) * batch["mask"][
            ..., None
        ]
        prediction = self.denoiser(
            torch.cat([noisy, local], -1), conditioning, timesteps
        )[..., :205]
        per_frame = (prediction - clean).square().mean(-1)
        reconstruction = (
            (per_frame * batch["mask"]).sum(1) / batch["mask"].sum(1)
        ).mean()
        endpoint = reconstruction.new_zeros(())
        if endpoint_weight:
            if skeleton is None:
                raise ValueError(
                    "Endpoint supervision requires an audited metric skeleton"
                )
            frames = batch["event_frames"]
            if frames.shape != (len(motion),) or frames.dtype != torch.long:
                raise ValueError("Event frames must be int64 [batch]")
            if torch.any((frames < 0) | (frames >= batch["mask"].sum(1))):
                raise ValueError("Event frame lies outside its valid source window")
            joints = skeleton(self.decode_motion(prediction))[..., :22, :]
            if target_frame == "fixed_world":
                wrists = joints[..., [20, 21], :]
            elif target_frame == "instantaneous_body":
                wrists = wrist_positions_in_body_frame(joints)
            else:
                raise ValueError("Unknown target frame")
            measured = wrists[
                torch.arange(len(motion), device=motion.device), frames, batch["hands"]
            ]
            endpoint = (measured - batch["positions"]).square().sum(-1).mean()
        metrics = {"reconstruction": reconstruction.detach()}
        if endpoint_weight:
            metrics["endpoint_squared_meters"] = endpoint.detach()
        return reconstruction + endpoint_weight * endpoint, metrics

    @torch.no_grad()
    def sample(
        self,
        text,
        local,
        local_mask,
        mask,
        hands,
        positions,
        seeds,
        steps=50,
        enabled=True,
    ):
        """Generate from requested conditions without reading a source motion.

        enabled=False bypasses all new residuals for a matched official-backbone
        baseline. Seeds are per sequence, so batching does not alter initial noise.
        This is deterministic DDIM without classifier-free guidance.
        """
        if not isinstance(steps, int) or not 2 <= steps <= len(self.alphas):
            raise ValueError("DDIM steps must be between 2 and the diffusion length")
        if len(seeds) != len(mask):
            raise ValueError("Exactly one noise seed is required per sequence")
        normalized_local, conditioning = self.conditioning(
            text, local, local_mask, mask, hands, positions, enabled
        )
        noise = (
            torch.stack(
                [
                    torch.randn(
                        mask.shape[1],
                        205,
                        device=local.device,
                        dtype=local.dtype,
                        generator=torch.Generator(device=local.device).manual_seed(
                            int(seed)
                        ),
                    )
                    for seed in seeds
                ]
            )
            * mask[..., None]
        )
        timeline = torch.linspace(len(self.alphas) - 1, 0, steps).long().tolist()
        for schedule_index, timestep in enumerate(timeline):
            timesteps = torch.full(
                (len(mask),), timestep, device=local.device, dtype=torch.long
            )
            predicted = self.denoiser(
                torch.cat([noise, normalized_local], -1), conditioning, timesteps
            )[..., :205]
            if schedule_index == len(timeline) - 1:
                return self.decode_motion(predicted) * mask[..., None]
            alpha = self.alphas[timestep]
            next_alpha = self.alphas[timeline[schedule_index + 1]]
            predicted_noise = (noise - alpha.sqrt() * predicted) / (1 - alpha).sqrt()
            noise = (
                next_alpha.sqrt() * predicted
                + (1 - next_alpha).sqrt() * predicted_noise
            ) * mask[..., None]

    def adapter_state(self):
        """Serialize only new parameters and target normalization, never the base."""
        return {
            name: value.detach().cpu().clone()
            for name, value in self.denoiser.state_dict().items()
            if not name.startswith("base.")
        }

    def load_adapter(self, state):
        expected = self.adapter_state()
        if set(state) != set(expected):
            raise ValueError("Checkpoint must contain exactly the reach adapter state")
        for name, value in state.items():
            if value.shape != expected[name].shape or not torch.isfinite(value).all():
                raise ValueError(f"Invalid adapter tensor: {name}")
        if torch.any(state["target_scale"] <= 0):
            raise ValueError("Checkpoint target scales must be positive")
        self.denoiser.load_state_dict(state, strict=False)
