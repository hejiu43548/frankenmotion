import torch
import numpy as np


@torch.no_grad()
def generate(model, local, text_embeddings, command_features, seed=0, steps=50):
    local = local[None] if local.ndim == 2 else local
    batch_size, num_frames, _ = local.shape
    device = local.device
    normalizer_input = torch.cat(
        [torch.zeros(batch_size, num_frames, 205, device=device), local], -1
    )
    local = model.motion_normalizer(normalizer_input)[..., 205:]
    conditioning = dict(
        mask=torch.ones(batch_size, num_frames, device=device, dtype=torch.bool),
        tx=model.prepare_tx_emb(text_embeddings),
    )
    noise = torch.randn(
        batch_size,
        num_frames,
        205,
        generator=torch.Generator(device=device).manual_seed(seed),
        device=device,
    )
    timeline = np.linspace(model.timesteps - 1, 0, steps, dtype=int)
    model.denoiser.static_residuals = model.denoiser.controller(command_features)
    try:
        for step_index, step in enumerate(timeline):
            noisy_motion = torch.cat([noise, local], -1)
            prediction = model.denoiser(
                noisy_motion,
                conditioning,
                torch.full((batch_size,), int(step), device=device, dtype=torch.long),
            )
            if step_index == len(timeline) - 1:
                return model.motion_normalizer.inverse(prediction)[..., :205]
            alpha = model.alphas_cumprod[step]
            next_alpha = model.alphas_cumprod[timeline[step_index + 1]]
            predicted_noise = (noisy_motion - alpha.sqrt() * prediction) / (
                1 - alpha
            ).sqrt()
            noise = (
                next_alpha.sqrt() * prediction
                + (1 - next_alpha).sqrt() * predicted_noise
            )[..., :205]
    finally:
        model.denoiser.static_residuals = None
