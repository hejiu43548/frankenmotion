import torch, numpy as np


@torch.no_grad()
def generate(model, local, tx, c, seed=0, steps=50):
    local = local[None] if local.ndim == 2 else local
    b, n, _ = local.shape
    device = local.device
    dummy = torch.cat([torch.zeros(b, n, 205, device=device), local], -1)
    local = model.motion_normalizer(dummy)[..., 205:]
    y = dict(
        mask=torch.ones(b, n, device=device, dtype=torch.bool),
        tx=model.prepare_tx_emb(tx),
    )
    noise = torch.randn(
        b,
        n,
        205,
        generator=torch.Generator(device=device).manual_seed(seed),
        device=device,
    )
    timeline = np.linspace(model.timesteps - 1, 0, steps, dtype=int)
    model.denoiser.static_residuals = model.denoiser.controller(c)
    try:
        for i, step in enumerate(timeline):
            x = torch.cat([noise, local], -1)
            pred = model.denoiser(
                x, y, torch.full((b,), int(step), device=device, dtype=torch.long)
            )
            if i == len(timeline) - 1:
                return model.motion_normalizer.inverse(pred)[..., :205]
            alpha = model.alphas_cumprod[step]
            next_alpha = model.alphas_cumprod[timeline[i + 1]]
            eps = (x - alpha.sqrt() * pred) / (1 - alpha).sqrt()
            noise = (next_alpha.sqrt() * pred + (1 - next_alpha).sqrt() * eps)[
                ..., :205
            ]
    finally:
        model.denoiser.static_residuals = None
