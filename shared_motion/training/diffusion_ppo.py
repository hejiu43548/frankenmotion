"""Stochastic DDIM transitions and PPO likelihood ratios for residual policies.

The policy action is the next noisy motion, not the final reward or a teacher's
motion. Gaussian likelihoods sum over valid motion dimensions; padding is absent
from the probability ratio. Final deterministic decoding has no policy likelihood.
"""

import torch


def transition(model, noisy, local, conditioning, timestep, next_timestep, eta):
    if not 0 < eta <= 1 or not 0 <= next_timestep < timestep < len(model.alphas):
        raise ValueError("Expected positive DDIM eta and decreasing positive step")
    timesteps = torch.full(
        (len(noisy),), timestep, device=noisy.device, dtype=torch.long
    )
    predicted = model.denoiser(torch.cat([noisy, local], -1), conditioning, timesteps)[
        ..., :205
    ]
    alpha = model.alphas[timestep]
    next_alpha = model.alphas[next_timestep]
    variance = eta**2 * (1 - next_alpha) / (1 - alpha) * (1 - alpha / next_alpha)
    predicted_noise = (noisy - alpha.sqrt() * predicted) / (1 - alpha).sqrt()
    mean = (
        next_alpha.sqrt() * predicted
        + (1 - next_alpha - variance).clamp_min(0).sqrt() * predicted_noise
    )
    return mean * conditioning["mask"][..., None], variance


def gaussian_log_ratio(action, mean, old_mean, variance, mask):
    if torch.any(variance <= 0):
        raise ValueError("Stochastic transition variance must be positive")
    # Difference before reduction avoids subtracting large Gaussian constants.
    new_error = (action.double() - mean.double()).square()
    old_error = (action.double() - old_mean.double()).square()
    return ((old_error - new_error) * mask[..., None]).sum(dim=(1, 2)) / (
        2 * variance.double()
    )


def clipped_policy_loss(log_ratio, advantages, clip_range=0.2):
    if not 0 < clip_range < 1:
        raise ValueError("PPO clip range must lie in (0,1)")
    ratio = log_ratio.clamp(-20, 20).exp()
    objective = torch.minimum(
        ratio * advantages, ratio.clamp(1 - clip_range, 1 + clip_range) * advantages
    )
    return -objective.mean(), ((ratio - 1).abs() > clip_range).float().mean()


@torch.no_grad()
def collect(model, batch, seeds, steps=20, eta=1.0):
    if not 2 <= steps <= len(model.alphas) or len(seeds) != len(batch["mask"]):
        raise ValueError("Invalid rollout schedule or seed count")
    local, conditioning = model.conditioning(
        batch["tx"],
        batch["local"],
        batch["local_mask"],
        batch["mask"],
        batch["hands"],
        batch["positions"],
    )
    generators = [
        torch.Generator(device=local.device).manual_seed(int(seed)) for seed in seeds
    ]
    noisy = (
        torch.stack(
            [
                torch.randn(
                    local.shape[1], 205, device=local.device, generator=generator
                )
                for generator in generators
            ]
        )
        * batch["mask"][..., None]
    )
    timeline = torch.linspace(len(model.alphas) - 1, 0, steps).long().tolist()
    trajectory = []
    for timestep, next_timestep in zip(timeline[:-1], timeline[1:]):
        mean, variance = transition(
            model, noisy, local, conditioning, timestep, next_timestep, eta
        )
        innovation = torch.stack(
            [
                torch.randn(
                    local.shape[1], 205, device=local.device, generator=generator
                )
                for generator in generators
            ]
        )
        action = (mean + variance.sqrt() * innovation) * batch["mask"][..., None]
        trajectory.append(
            dict(
                noisy=noisy,
                action=action,
                old_mean=mean,
                timestep=timestep,
                next_timestep=next_timestep,
            )
        )
        noisy = action
    timesteps = torch.zeros(len(noisy), device=noisy.device, dtype=torch.long)
    predicted = model.denoiser(torch.cat([noisy, local], -1), conditioning, timesteps)[
        ..., :205
    ]
    return model.decode_motion(predicted) * batch["mask"][..., None], trajectory
