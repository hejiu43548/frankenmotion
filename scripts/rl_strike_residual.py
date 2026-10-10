"""Online PPO for a compact post-generation arm residual; no teacher targets."""

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
from shared_motion.training.geometry import Skeleton
from shared_motion.training.model import file_sha256
from shared_motion.training.reach3d_model import ReachDiffusion
from shared_motion.training.reach3d_geometry import wrist_positions_in_body_frame
from shared_motion.training.strike_residual_policy import (
    StrikeResidualPolicy,
    observations,
    apply_residual,
)
from shared_motion.training.diffusion_ppo import clipped_policy_loss


@hydra.main(
    version_base="1.3", config_path="../config", config_name="rl_strike_residual"
)
def main(config):
    torch.set_num_threads(4)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    sampler = np.random.default_rng(config.seed)
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
        raise ValueError("Backbone mismatch")
    model.load_adapter(checkpoint["adapter"])
    model.requires_grad_(False)
    skeleton = Skeleton(training.skeleton).to(config.device)
    rows, items = load_items(training.data, "train")
    targets = np.array([row["target_xyz_m"] for row in rows])
    policy = StrikeResidualPolicy().to(config.device)
    optimizer = torch.optim.Adam(policy.parameters(), lr=config.learning_rate)
    output = Path(config.output)
    if (output / "metrics.jsonl").exists():
        raise ValueError("Existing run must not be overwritten")
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output / "config.yaml", resolve=True)
    started = time.monotonic()
    for iteration in range(1, config.iterations + 1):
        indices = sampler.integers(
            0, len(items), size=config.conditions_per_batch
        ).tolist()
        # Match durations within a batch to avoid padding-dependent residual windows.
        selected_frames = len(items[indices[0]]["motion"])
        eligible = [
            index
            for index, item in enumerate(items)
            if len(item["motion"]) == selected_frames
        ]
        indices = sampler.choice(eligible, size=config.conditions_per_batch).tolist()
        batch = make_batch(items, indices, config.device)
        requested = sampler.uniform(
            targets.min(0), targets.max(0), size=(config.conditions_per_batch, 3)
        )
        batch["positions"] = torch.tensor(
            requested, device=config.device, dtype=torch.float32
        )
        seeds = sampler.integers(0, 2**31, size=config.conditions_per_batch).tolist()
        with torch.no_grad():
            base = model.sample(
                batch["tx"],
                batch["local"],
                batch["local_mask"],
                batch["mask"],
                batch["hands"],
                batch["positions"],
                seeds,
                steps=config.ddim_steps,
            )
            state = observations(
                base, batch["positions"], batch["event_frames"], skeleton
            )
            expanded_state = state.repeat_interleave(config.actions_per_condition, 0)
            distribution = policy.distribution(expanded_state)
            actions = distribution.sample()
            old_log_probability = distribution.log_prob(actions).sum(-1)
            expanded_motion = base.repeat_interleave(config.actions_per_condition, 0)
            expanded_frames = batch["event_frames"].repeat_interleave(
                config.actions_per_condition
            )
            expanded_targets = batch["positions"].repeat_interleave(
                config.actions_per_condition, 0
            )
            corrected = apply_residual(
                expanded_motion,
                actions,
                expanded_frames,
                skeleton,
                config.max_angle,
                config.radius_frames,
            )
            joints = skeleton(corrected)
            wrists = wrist_positions_in_body_frame(joints[..., :22, :])
            measured = wrists[
                torch.arange(len(corrected), device=config.device), expanded_frames, 1
            ]
            errors = (measured - expanded_targets).norm(dim=-1)
            reward = torch.exp(-(errors / config.reward_scale_m).square())
            grouped = reward.reshape(
                config.conditions_per_batch, config.actions_per_condition
            )
            advantage = (
                (grouped - grouped.mean(1, keepdim=True))
                / grouped.std(1, keepdim=True, unbiased=False).clamp_min(1e-4)
            ).flatten()
        for update_epoch in range(config.update_epochs):
            log_probability = (
                policy.distribution(expanded_state).log_prob(actions).sum(-1)
            )
            loss, clipped_fraction = clipped_policy_loss(
                log_probability - old_log_probability, advantage, config.clip_range
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            gradient_norm = torch.nn.utils.clip_grad_norm_(
                policy.parameters(), 1.0, error_if_nonfinite=True
            )
            optimizer.step()
        if iteration == 1 or iteration % 20 == 0:
            with torch.no_grad():
                deterministic = apply_residual(
                    base,
                    policy.mean(state),
                    batch["event_frames"],
                    skeleton,
                    config.max_angle,
                    config.radius_frames,
                )
                wrists = wrist_positions_in_body_frame(
                    skeleton(deterministic)[..., :22, :]
                )
                measured = wrists[
                    torch.arange(len(base), device=config.device),
                    batch["event_frames"],
                    1,
                ]
                final_error = (measured - batch["positions"]).norm(dim=-1).mean()
            record = dict(
                iteration=iteration,
                reward=float(reward.mean()),
                sampled_error_m=float(errors.mean()),
                deterministic_error_m=float(final_error),
                clipped_fraction=float(clipped_fraction),
                gradient_norm=float(gradient_norm),
                seconds=time.monotonic() - started,
            )
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
    torch.save(
        dict(
            format="strike_arm_ppo_v1",
            policy=policy.state_dict(),
            optimizer=optimizer.state_dict(),
            config=OmegaConf.to_container(config, resolve=True),
            initial_sha256=file_sha256(config.initial),
            backbone_sha256=model.backbone_sha256,
            sampler_state=sampler.bit_generator.state,
            torch_rng=torch.get_rng_state(),
            cuda_rng=torch.cuda.get_rng_state_all(),
        ),
        output / "final.pt",
    )
    (output / "completion.json").write_text(
        json.dumps(
            dict(
                iterations=config.iterations,
                seconds=time.monotonic() - started,
                sha256=file_sha256(output / "final.pt"),
                teacher_used=False,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
