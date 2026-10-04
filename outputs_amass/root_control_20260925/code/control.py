"""Root speed (m/s) and body-yaw-rate (rad/s) adapters for FrankenMotion."""
import torch
from torch import nn

FPS = 20.0
WINDOW = 20

def root_signals(raw):
    return torch.stack((torch.linalg.vector_norm(raw[..., 1:3], dim=-1)*FPS,
                        raw[..., 3]*FPS), -1)

def window_profile(values, valid):
    out = torch.zeros_like(values)
    for start in range(0, values.shape[1], WINDOW):
        end = min(start + WINDOW, values.shape[1])
        mask = valid[:, start:end, None].to(values)
        mean = (values[:, start:end]*mask).sum(1, keepdim=True)/mask.sum(1, keepdim=True).clamp_min(1)
        out[:, start:end] = mean
    return out

def labels(raw, lengths):
    # Last increment of a sequence can be extrapolated; always exclude it.
    valid = torch.arange(raw.shape[1], device=raw.device)[None] < (lengths[:, None]-1)
    values = window_profile(root_signals(raw), valid)
    return values, valid

def encode_control(values, valid):
    if valid.ndim == 2:
        valid = valid[..., None].expand_as(values)
    scales = values.new_tensor([3.0, 3.141592653589793])
    return torch.cat((values/scales*valid, valid.to(values)), -1)

class RootControl(nn.Module):
    """Frozen backbone; zero-initialized residuals after each transformer block.

    control[..., :2] physical speed/yaw rate scaled by (3 m/s, pi rad/s);
    control[..., 2:] availability masks. Missing control is exact base behavior.
    """
    def __init__(self, base):
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad_(False)
        dim = base.latent_dim
        self.encoder = nn.Sequential(nn.Linear(4, 128), nn.SiLU(), nn.Linear(128, dim))
        self.residuals = nn.ModuleList([
            nn.Sequential(nn.Linear(dim, 128), nn.SiLU(), nn.Linear(128, dim))
            for _ in base.seqTransEncoder.layers])
        for branch in self.residuals:
            nn.init.zeros_(branch[-1].weight)
            nn.init.zeros_(branch[-1].bias)
        self._condition = None
        self._handles = [layer.register_forward_hook(self._hook(i))
                         for i, layer in enumerate(base.seqTransEncoder.layers)]

    def _hook(self, index):
        def apply(module, args, output):
            if self._condition is None:
                return output
            features, gate = self._condition
            residual = self.residuals[index](features)*gate
            prefix = output.shape[1]-residual.shape[1]
            return output + torch.nn.functional.pad(residual, (0, 0, prefix, 0))
        return apply

    def forward(self, x, y, t, tf=None):
        control = y.get('root_control')
        self._condition = None if control is None else (
            self.encoder(control), control[..., 2:].amax(-1, keepdim=True))
        try:
            return self.base(x, y, t, tf)
        finally:
            self._condition = None

    def train(self, mode=True):
        super().train(mode)
        self.base.eval()  # Preserve frozen backbone/dropout behavior.
        return self

    def adapter_state(self):
        return {k: v for k, v in self.state_dict().items() if not k.startswith('base.')}

    def load_adapter(self, state):
        result = self.load_state_dict(state, strict=False)
        assert not result.unexpected_keys
        assert all(k.startswith('base.') for k in result.missing_keys)
