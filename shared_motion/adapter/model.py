"""Load exactly one selected shared20 checkpoint, with no legacy imports."""

import copy
import torch
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

    def forward(self, values):
        return values if self.disable else (values - self.mean) / (self.std + self.eps)

    def inverse(self, values):
        return values if self.disable else values * self.std + self.mean


def load(path, device="cpu"):
    import src.prepare
    from hydra.utils import instantiate
    from omegaconf import OmegaConf

    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    if (
        checkpoint["kind"] != "frankenmotion_shared20_v1"
        or checkpoint["tasks"] != TASKS
    ):
        raise ValueError("Expected final 55-input shared20 checkpoint")
    if checkpoint["fields"] != FIELDS or checkpoint["intents"] != INTENTS:
        raise ValueError("Command ABI mismatch")
    config = copy.deepcopy(checkpoint["diffusion_config"])
    for name in ["motion_normalizer", "text_normalizer"]:
        config[name]["_target_"] = "shared_motion.adapter.model.EmbeddedNormalizer"
    model = instantiate(OmegaConf.create(config))
    model.denoiser = UnifiedControl(model.denoiser, SharedCommands(checkpoint["width"]))
    model.load_state_dict(checkpoint["state_dict"], strict=True)
    return model.to(device).eval(), checkpoint


def save(model, checkpoint, path, step):
    output = dict(
        checkpoint,
        step=step,
        state_dict={
            parameter_name: parameter_value.detach().cpu()
            for parameter_name, parameter_value in model.state_dict().items()
        },
    )
    torch.save(output, path)


def legacy_features(
    kind, value, frames, extra=None, skeleton_height=1.372592926, device="cpu"
):
    command_value = torch.tensor([value], device=device, dtype=torch.float32)
    phase = torch.linspace(0, 1, frames, device=device)[None]
    extra_features = (
        None
        if extra is None
        else torch.tensor([extra], device=device, dtype=torch.float32)
    )
    if kind == "root_profile":
        root_velocity = torch.stack(
            [
                extra_features[:, 0, None]
                + phase * (extra_features[:, 1] - extra_features[:, 0])[:, None],
                extra_features[:, 2, None]
                + phase * (extra_features[:, 3] - extra_features[:, 2])[:, None],
            ],
            -1,
        )
        extra_features = encode_control(
            root_velocity, extra_features[:, None, 4:6].expand(1, frames, 2)
        )
    command_features = make(
        kind,
        command_value,
        phase,
        extra_features,
        frames=frames,
        skeleton_height=skeleton_height,
    )
    if kind == "walk_endpoint":
        frame_times = torch.arange(frames, device=device) * 0.05
        speed_profile = torch.clamp((frame_times - 0.6) / 0.6, 0, 1) * torch.clamp(
            (5.3 - frame_times) / 0.8, 0, 1
        )
        speed_profile /= speed_profile[:-1].sum() * 0.05
        root_velocity = torch.zeros(1, frames, 2, device=device)
        root_velocity[:, :, 0] = extra_features[:, 0, None] * speed_profile
        root_velocity[:, :, 1] = (
            extra_features[:, 1, None] * (frame_times < 1.2).float() / 1.2
        )
        command_features[:, :, 18:22] = encode_control(
            root_velocity, torch.ones(1, frames, dtype=torch.bool, device=device)
        )
    return command_features


def command(task, value, frames, device="cpu", extra=None):
    phase = torch.linspace(0, 1, frames, device=device)[None]
    value = torch.as_tensor(value, device=device, dtype=torch.float32).reshape(1)
    if task not in NEW:
        if task == "place_hold_retract" and isinstance(extra, list):
            extra = extra[0]
        return torch.nn.functional.pad(
            legacy_features(task, float(value), frames, extra, device=device), (0, 9)
        )
    command_features = torch.nn.functional.pad(
        make("root_profile", value, phase, frames=frames), (0, 9)
    )
    command_features[:, :, 11] = 0
    command_features[:, :, 46 + list(NEW).index(task)] = 1
    lower_bound, upper_bound = NEW[task]["bounds"]
    command_features[:, :, 16] = (
        2 * (value - lower_bound) / (upper_bound - lower_bound) - 1
    )
    command_features[:, :, 17] = 1
    return command_features
