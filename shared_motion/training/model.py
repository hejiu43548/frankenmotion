"""Hydra-loaded official backbone and the two explicitly distinct controllers."""

import hashlib
from pathlib import Path

import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from torch import nn

from shared_motion.adapter.inputs import encode_control, make
from shared_motion.adapter.model import command
from shared_motion.adapter.network import SharedCommands, UnifiedControl
from .adapters import RootControl, TaskControl
from .reach import REACH_INDEX, REACH_POLICY, command_vectors, normalized_commands
from .turn import TURN_NATIVE_SPEED, TURN_POLICY, TURN_RANGE
from .catalog import COMMAND_RANGES, HUMAN_HEIGHT, ROOT_TASKS, TASK_NAMES


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_official_backbone(
    checkpoint, denoiser, schedule, timesteps=100, expected_sha256=None
):
    checkpoint = Path(checkpoint).expanduser().resolve()
    digest = file_sha256(checkpoint)
    if expected_sha256 is not None and digest != expected_sha256:
        raise ValueError(f"Backbone SHA256 mismatch: {checkpoint}")
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)["state_dict"]
    denoiser.load_state_dict(
        {
            name.removeprefix("denoiser."): value
            for name, value in state.items()
            if name.startswith("denoiser.")
        },
        strict=True,
    )
    denoiser.requires_grad_(False)
    return dict(
        denoiser=denoiser,
        mean=state["motion_normalizer.mean"],
        std=state["motion_normalizer.std"],
        text_mean=state["text_normalizer.mean"],
        text_std=state["text_normalizer.std"],
        alphas=(1 - schedule(timesteps)).cumprod(0).float(),
        sha256=digest,
    )


def create_charlie_root(base, phase, bottleneck=128, embedding_width=32):
    root = RootControl(base, bottleneck)
    return (
        root
        if phase == "root"
        else TaskControl(root, COMMAND_RANGES, bottleneck, embedding_width)
    )


def create_shared_commands(base, phase, width=1024, zero_initialize=True):
    if base.latent_dim != 512 or len(base.seqTransEncoder.layers) != 4:
        raise ValueError(
            "The release SharedCommands requires four512-wide backbone layers"
        )
    controller = SharedCommands(width)
    # Append XYZ fields without changing the legacy release controller class.
    controller.encoder[0] = nn.Linear(58, width)
    if zero_initialize:
        for head in [*controller.residuals, controller.output]:
            nn.init.zeros_(head[-1].weight)
            nn.init.zeros_(head[-1].bias)
    return UnifiedControl(base, controller)


def root_signals(motion):
    return torch.stack([motion[..., 1:3].norm(dim=-1) * 20, motion[..., 3] * 20], -1)


def smooth_profile(values, valid, window=20):
    output = torch.zeros_like(values)
    for start in range(0, values.shape[1], window):
        stop = min(start + window, values.shape[1])
        mask = valid[:, start:stop, None].to(values)
        average = (values[:, start:stop] * mask).sum(1, keepdim=True) / mask.sum(
            1, keepdim=True
        ).clamp_min(1)
        output[:, start:stop] = average
    return output


def training_root(batch, dropout=0.0, only_supported=False):
    valid = (
        torch.arange(batch["motion"].shape[1], device=batch["motion"].device)[None]
        < batch["lengths"][:, None] - 1
    )
    values = smooth_profile(root_signals(batch["motion"]), valid)
    available = valid[..., None].expand_as(values).clone()
    if only_supported:
        available &= torch.isin(batch["task"], batch["task"].new_tensor(ROOT_TASKS))[
            :, None, None
        ]
    if dropout:
        available &= torch.rand(len(values), 1, 2, device=values.device) > dropout
    return values, valid, available, encode_control(values, available)


class ControlledDiffusion(nn.Module):
    def __init__(self, bundle, controller_config, phase):
        super().__init__()
        self.phase = phase
        self.task_policy = dict(turn=TURN_POLICY, reach=REACH_POLICY)
        self.kind = (
            "charlie_root"
            if str(controller_config._target_).endswith("create_charlie_root")
            else "shared_commands"
        )
        self.denoiser = instantiate(
            controller_config, base=bundle["denoiser"], phase=phase
        )
        self.backbone_sha256 = bundle["sha256"]
        for name in ["mean", "std", "text_mean", "text_std", "alphas"]:
            self.register_buffer(name, bundle[name].clone())
        self.train(False)

    @property
    def backbone(self):
        if self.kind == "shared_commands" or self.phase == "root":
            return self.denoiser.base
        return self.denoiser.root.base

    @property
    def root_branch(self):
        if self.kind != "charlie_root":
            return None
        return self.denoiser if self.phase == "root" else self.denoiser.root

    def adapter_state(self):
        return {
            name: value.detach().cpu().clone()
            for name, value in self.denoiser.state_dict().items()
            if not name.startswith(("base.", "root.base."))
        }

    def load_adapter(self, state, root_only=False):
        target = (
            self.root_branch
            if root_only and self.kind == "charlie_root"
            else self.denoiser
        )
        result = target.load_state_dict(state, strict=False)
        if result.unexpected_keys or any(
            not name.startswith(("base.", "root.base.")) for name in result.missing_keys
        ):
            raise ValueError(f"Incompatible controller checkpoint: {result}")

    def normalize(self, features):
        return (features - self.mean) / (self.std + 1e-12)

    def decode(self, features):
        return features * (self.std + 1e-12) + self.mean

    def inputs(self, batch):
        local = (batch["local"] - self.mean[205:]) / (self.std[205:] + 1e-12)
        local = local * batch["local_mask"] * batch["mask"][..., None]
        clean = (
            torch.cat(
                [(batch["motion"] - self.mean[:205]) / (self.std[:205] + 1e-12), local],
                -1,
            )
            * batch["mask"][..., None]
        )
        text = (batch["tx"] - self.text_mean) / (self.text_std + 1e-12)
        conditioning = dict(
            mask=batch["mask"],
            tx=dict(
                x=text[:, None],
                mask=torch.ones(len(text), 1, device=text.device, dtype=torch.bool),
            ),
        )
        return clean, conditioning

    def requested_root(self, batch, commands, skeleton):
        commands = command_vectors(commands, batch["task"])
        scalar_commands = commands[:, 0]
        values = commands.new_zeros(len(commands), batch["mask"].shape[1], 2)
        supported = torch.isin(batch["task"], batch["task"].new_tensor(ROOT_TASKS))
        valid = batch["mask"] & supported[:, None]
        turning = batch["task"] == TASK_NAMES.index("turn")
        values[turning, :, 0] = TURN_NATIVE_SPEED
        values[turning, :, 1] = -scalar_commands[turning, None] / (
            (batch["lengths"][turning, None] - 1) / 20
        )
        walking = supported & ~turning
        values[walking, :, 0] = (
            scalar_commands[walking, None] * skeleton.height / HUMAN_HEIGHT
        )
        return encode_control(values, valid)

    def unified_features(self, batch, commands, root_control, use_task):
        commands = command_vectors(commands, batch["task"])
        normalized = normalized_commands(commands, batch["task"], COMMAND_RANGES)
        frames = batch["mask"].shape[1]
        rows = []
        for sample_index, task_index in enumerate(batch["task"].tolist()):
            length = int(batch["lengths"][sample_index])
            if use_task:
                features = command(
                    TASK_NAMES[task_index],
                    float(commands[sample_index, 0]),
                    length,
                    commands.device,
                )
                if TASK_NAMES[task_index] == "turn":
                    features[:, :, 16] = (
                        2
                        * (commands[sample_index, 0] - TURN_RANGE[0])
                        / (TURN_RANGE[1] - TURN_RANGE[0])
                        - 1
                    )
                features[:, :, 18:22] = root_control[
                    sample_index : sample_index + 1, :length
                ]
            else:
                phase = torch.linspace(0, 1, length, device=commands.device)[None]
                features = torch.nn.functional.pad(
                    make(
                        "root_profile",
                        commands[sample_index : sample_index + 1, 0],
                        phase,
                        root_control[sample_index : sample_index + 1, :length],
                        frames=length,
                    ),
                    (0, 9),
                )
            spatial = features.new_zeros(1, features.shape[1], 3)
            if use_task and task_index == REACH_INDEX:
                features[:, :, 16] = 0
                features[:, :, 17] = 1
                spatial[:] = normalized[sample_index]
            features = torch.cat([features, spatial], dim=-1)
            rows.append(torch.nn.functional.pad(features, (0, 0, 0, frames - length)))
        return torch.cat(rows)

    def predict(
        self,
        motion,
        conditioning,
        timesteps,
        batch,
        commands,
        root_control,
        mode="task",
    ):
        conditioning = dict(conditioning)
        if self.kind == "shared_commands":
            if self.denoiser.static_residuals is not None:
                return self.denoiser(motion, conditioning, timesteps)
            features = (
                None
                if mode == "backbone"
                else self.unified_features(
                    batch, commands, root_control, mode == "task"
                )
            )
            self.denoiser.command_features = features
            try:
                return self.denoiser(motion, conditioning, timesteps)
            finally:
                self.denoiser.command_features = None
        if mode != "backbone":
            conditioning["root_control"] = root_control
        if mode == "task" and self.phase != "root":
            conditioning["task_control"] = (batch["task"], commands)
        return self.denoiser(motion, conditioning, timesteps)

    def controller_outputs(self, batch, commands, skeleton):
        if self.kind == "charlie_root":
            return (self.denoiser.outputs(batch["task"], commands),)
        root_control = self.requested_root(batch, commands, skeleton)
        return self.denoiser.controller(
            self.unified_features(batch, commands, root_control, True)
        )

    def sample(self, batch, commands, skeleton, seeds, steps=50, mode="task"):
        # Generation only reads text, masks, lengths, task IDs and requested commands.
        local = (batch["local"] - self.mean[205:]) / (self.std[205:] + 1e-12)
        local = local * batch["local_mask"] * batch["mask"][..., None]
        text = (batch["tx"] - self.text_mean) / (self.text_std + 1e-12)
        conditioning = dict(
            mask=batch["mask"],
            tx=dict(
                x=text[:, None],
                mask=torch.ones(len(text), 1, device=text.device, dtype=torch.bool),
            ),
        )
        noise = torch.stack(
            [
                torch.randn(
                    local.shape[1],
                    205,
                    device=local.device,
                    generator=torch.Generator(device=local.device).manual_seed(
                        int(seed)
                    ),
                )
                for seed in seeds
            ]
        )
        timeline = np.linspace(len(self.alphas) - 1, 0, steps, dtype=int).tolist()
        root_control = self.requested_root(batch, commands, skeleton)
        if not 2 <= steps <= len(self.alphas):
            raise ValueError("DDIM steps must be between2 and backbone timesteps")
        if self.kind == "shared_commands" and mode != "backbone":
            features = self.unified_features(
                batch, commands, root_control, mode == "task"
            )
            self.denoiser.static_residuals = self.denoiser.controller(features)
        try:
            for schedule_index, timestep in enumerate(timeline):
                predicted = self.predict(
                    torch.cat([noise, local], -1),
                    conditioning,
                    torch.full(
                        (len(noise),), timestep, device=noise.device, dtype=torch.long
                    ),
                    batch,
                    commands,
                    root_control,
                    mode,
                )[..., :205]
                if schedule_index == len(timeline) - 1:
                    return predicted * (self.std[:205] + 1e-12) + self.mean[:205]
                alpha = self.alphas[timestep]
                next_alpha = self.alphas[timeline[schedule_index + 1]]
                noise = (
                    next_alpha.sqrt() * predicted
                    + (1 - next_alpha).sqrt()
                    * (noise - alpha.sqrt() * predicted)
                    / (1 - alpha).sqrt()
                )
        finally:
            if self.kind == "shared_commands":
                self.denoiser.static_residuals = None


def build_model(config, phase):
    bundle = instantiate(config.backbone)
    model = ControlledDiffusion(bundle, config.controller, phase)
    model.backbone_configuration = OmegaConf.to_container(config.backbone, resolve=True)
    return model
