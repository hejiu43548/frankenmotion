"""Online diffusion PPO on new XYZ requests, with real-motion replay.

No external teacher, old specialist or generated target motion is loaded. Rewards
are evaluated on the current policy's sampled motions. The initial residual is
this experiment's new supervised adapter; the official backbone remains frozen.
"""

import json
from pathlib import Path
import sys
import time

import hydra
from hydra.utils import instantiate
import numpy as np
from omegaconf import OmegaConf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_strike3d import load_items, make_batch
from shared_motion.training.diffusion_ppo import (
    collect,
    transition,
    gaussian_log_ratio,
    clipped_policy_loss,
)
from shared_motion.training.geometry import Skeleton
from shared_motion.training.model import file_sha256
from shared_motion.training.reach3d_geometry import wrist_positions_in_body_frame
from shared_motion.training.reach3d_model import ReachDiffusion


@hydra.main(version_base="1.3", config_path="../config", config_name="rl_strike3d")
def main(config):
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    sampler = np.random.default_rng(config.seed)
    generator = torch.Generator(device=config.device).manual_seed(config.seed + 1)
    checkpoint = torch.load(config.initial, map_location="cpu", weights_only=False)
    training = OmegaConf.create(checkpoint["config"])
    audit = checkpoint["audit"]
    model = (
        ReachDiffusion(
            instantiate(training.backbone),
            audit["train_target_center"],
            audit["train_target_scale"],
        )
        .to(config.device)
        .eval()
    )
    if model.backbone_sha256 != checkpoint["backbone_sha256"]:
        raise ValueError("Initial backbone mismatch")
    model.load_adapter(checkpoint["adapter"])
    skeleton = Skeleton(training.skeleton).to(config.device)
    rows, items = load_items(training.data, "train")
    training_targets = np.array([row["target_xyz_m"] for row in rows])
    lower, upper = training_targets.min(0), training_targets.max(0)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=0.0,
    )
    output = Path(config.output)
    if (output / "metrics.jsonl").exists():
        raise ValueError("Refusing to overwrite an RL run")
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output / "rl_config.yaml", resolve=True)
    started = time.monotonic()
    for iteration in range(1, config.iterations + 1):
        for _ in range(1000):
            target = sampler.uniform(lower, upper)
            nearest_distance = np.linalg.norm(training_targets - target, axis=1).min()
            if nearest_distance >= config.minimum_target_separation_m:
                break
        else:
            raise ValueError("No supported target hole could be sampled")
        source_index = int(sampler.integers(len(items)))
        batch = make_batch(items, [source_index] * config.group_size, config.device)
        batch["positions"] = torch.tensor(
            target, device=config.device, dtype=torch.float32
        )[None].repeat(config.group_size, 1)
        seeds = sampler.integers(0, 2**31, size=config.group_size).tolist()
        motion, trajectory = collect(model, batch, seeds, config.ddim_steps, config.eta)
        with torch.no_grad():
            joints = skeleton(motion)
            wrists = wrist_positions_in_body_frame(joints[..., :22, :])
            measured = wrists[
                torch.arange(len(motion), device=config.device),
                batch["event_frames"],
                1,
            ]
            errors = (measured - batch["positions"]).norm(dim=-1)
            root_excursion = (
                (joints[:, :, 0] - joints[:, :1, 0]).norm(dim=-1).max(dim=1).values
            )
            peak_speed = (
                ((joints[:, 1:, 21] - joints[:, :-1, 21]) * 20)
                .norm(dim=-1)
                .max(dim=1)
                .values
            )
            valid = (
                (root_excursion <= config.maximum_root_excursion_m)
                & (peak_speed >= config.minimum_peak_wrist_speed)
                & (peak_speed <= config.maximum_peak_wrist_speed)
            )
            rewards = torch.exp(-(errors / config.reward_scale_m).square()) * valid
            advantages = (rewards - rewards.mean()) / rewards.std(
                unbiased=False
            ).clamp_min(1e-4)
        selected = trajectory[int(sampler.integers(len(trajectory)))]
        local, conditioning = model.conditioning(
            batch["tx"],
            batch["local"],
            batch["local_mask"],
            batch["mask"],
            batch["hands"],
            batch["positions"],
        )
        for update_epoch in range(config.update_epochs):
            mean, variance = transition(
                model,
                selected["noisy"],
                local,
                conditioning,
                selected["timestep"],
                selected["next_timestep"],
                config.eta,
            )
            log_ratio = gaussian_log_ratio(
                selected["action"], mean, selected["old_mean"], variance, batch["mask"]
            )
            policy_loss, clipped_fraction = clipped_policy_loss(
                log_ratio, advantages, config.clip_range
            )
            replay_batch = make_batch(
                items, sampler.integers(0, len(items), size=2).tolist(), config.device
            )
            replay_loss, _ = model.supervised_loss(
                replay_batch, generator, skeleton, config.endpoint_weight
            )
            loss = (
                policy_loss / config.policy_gradient_scale
                + config.replay_weight * replay_loss
            )
            optimizer.zero_grad(set_to_none=True)
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite PPO/replay loss")
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in model.parameters()
                    if parameter.requires_grad
                ],
                1.0,
                error_if_nonfinite=True,
            )
            optimizer.step()
        record = dict(
            iteration=iteration,
            reward=float(rewards.mean()),
            mean_error_m=float(errors.mean()),
            quality_gate_fraction=float(valid.float().mean()),
            nonzero_advantage=bool(advantages.abs().sum() > 0),
            policy_loss=float(policy_loss.detach()),
            replay_loss=float(replay_loss.detach()),
            clipped_fraction=float(clipped_fraction),
            last_epoch_log_ratio_max=float(log_ratio.detach().abs().max()),
            gradient_norm=float(gradient_norm),
            target=target.tolist(),
            nearest_training_target_m=float(nearest_distance),
            source_key=rows[source_index]["key"],
            seconds=time.monotonic() - started,
        )
        with (output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(record) + "\n")
        if iteration == 1 or iteration % 20 == 0:
            print(json.dumps(record), flush=True)
    checkpoint.update(
        adapter=model.adapter_state(),
        rl_config=OmegaConf.to_container(config, resolve=True),
        rl_iterations=config.iterations,
        initial_sha256=file_sha256(config.initial),
        optimizer=optimizer.state_dict(),
        sampler_state=sampler.bit_generator.state,
        generator_state=generator.get_state(),
    )
    torch.save(checkpoint, output / "final.pt")
    (output / "completion.json").write_text(
        json.dumps(
            dict(
                iterations=config.iterations,
                seconds=time.monotonic() - started,
                final_sha256=file_sha256(output / "final.pt"),
                teacher_used=False,
                backbone_sha256=model.backbone_sha256,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
