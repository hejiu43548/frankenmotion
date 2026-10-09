"""Load exactly one selected shared20 checkpoint, with no legacy imports."""

import copy, torch
from torch import nn
from .network import SharedCommands, UnifiedControl
from .schema import TASKS as ORIGINAL_TASKS, FIELDS, INTENTS
from .catalog import NEW
from .inputs import make, encode_control

TASKS = ORIGINAL_TASKS + list(NEW)


class EmbeddedNormalizer(nn.Module):
    def __init__(self, shape, eps=1e-12, disable=False):
        super().__init__()
        self.eps = eps
        self.disable = disable
        self.register_buffer("mean", torch.zeros(tuple(shape)))
        self.register_buffer("std", torch.ones(tuple(shape)))

    def forward(self, x):
        return x if self.disable else (x - self.mean) / (self.std + self.eps)

    def inverse(self, x):
        return x if self.disable else x * self.std + self.mean


def load(path, device="cpu"):
    import src.prepare
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    pack = torch.load(path, map_location="cpu", weights_only=False)
    if pack["kind"] != "frankenmotion_shared20_v1" or pack["tasks"] != TASKS:
        raise ValueError("Expected final 55-input shared20 checkpoint")
    if pack["fields"] != FIELDS or pack["intents"] != INTENTS:
        raise ValueError("Command ABI mismatch")
    config = copy.deepcopy(pack["diffusion_config"])
    for name in ["motion_normalizer", "text_normalizer"]:
        config[name]["_target_"] = "shared_motion.adapter.model.EmbeddedNormalizer"
    model = instantiate(OmegaConf.create(config))
    model.denoiser = UnifiedControl(model.denoiser, SharedCommands(pack["width"]))
    model.load_state_dict(pack["state_dict"], strict=True)
    return model.to(device).eval(), pack


def save(model, pack, path, step):
    output = dict(
        pack,
        step=step,
        state_dict={k: v.detach().cpu() for k, v in model.state_dict().items()},
    )
    torch.save(output, path)


def legacy_features(
    kind, value, frames, extra=None, fkheight=1.372592926, device="cpu"
):
    cmd = torch.tensor([value], device=device, dtype=torch.float32)
    phase = torch.linspace(0, 1, frames, device=device)[None]
    ext = (
        None
        if extra is None
        else torch.tensor([extra], device=device, dtype=torch.float32)
    )
    if kind == "root_profile":
        v = torch.stack(
            [
                ext[:, 0, None] + phase * (ext[:, 1] - ext[:, 0])[:, None],
                ext[:, 2, None] + phase * (ext[:, 3] - ext[:, 2])[:, None],
            ],
            -1,
        )
        ext = encode_control(v, ext[:, None, 4:6].expand(1, frames, 2))
    c = make(kind, cmd, phase, ext, frames=frames, fkheight=fkheight)
    if kind == "walk_endpoint":
        tt = torch.arange(frames, device=device) * 0.05
        pr = torch.clamp((tt - 0.6) / 0.6, 0, 1) * torch.clamp((5.3 - tt) / 0.8, 0, 1)
        pr /= pr[:-1].sum() * 0.05
        v = torch.zeros(1, frames, 2, device=device)
        v[:, :, 0] = ext[:, 0, None] * pr
        v[:, :, 1] = ext[:, 1, None] * (tt < 1.2).float() / 1.2
        c[:, :, 18:22] = encode_control(
            v, torch.ones(1, frames, dtype=torch.bool, device=device)
        )
    return c


def command(task, value, frames, device="cpu", extra=None):
    phase = torch.linspace(0, 1, frames, device=device)[None]
    value = torch.as_tensor(value, device=device, dtype=torch.float32).reshape(1)
    if task not in NEW:
        if task == "place_hold_retract" and isinstance(extra, list):
            extra = extra[0]
        return torch.nn.functional.pad(
            legacy_features(task, float(value), frames, extra, device=device), (0, 9)
        )
    c = torch.nn.functional.pad(
        make("root_profile", value, phase, frames=frames), (0, 9)
    )
    c[:, :, 11] = 0
    c[:, :, 46 + list(NEW).index(task)] = 1
    lo, hi = NEW[task]["bounds"]
    c[:, :, 16] = 2 * (value - lo) / (hi - lo) - 1
    c[:, :, 17] = 1
    return c
