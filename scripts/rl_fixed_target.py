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
from shared_motion.training.fixed_target import (
    FixedTargetPolicy,
    fixed_observations,
    apply_fixed_residual,
    fixed_metrics,
    selected_wrists,
)
from shared_motion.training.diffusion_ppo import clipped_policy_loss


@hydra.main(version_base="1.3", config_path="../config", config_name="rl_fixed_target")
def main(config):
    torch.set_num_threads(4)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    sampler = np.random.default_rng(config.seed)
    checkpoint = torch.load(config.initial, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "fixed_world_adapter_v1":
        raise ValueError(
            "Expected fixed-world adapter, not legacy moving-target weights"
        )
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
    policy = FixedTargetPolicy().to(config.device)
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
        requested = []
        for item_index in indices:
            hand = int(items[item_index]["hands"])
            support = targets[[int(item["hands"]) == hand for item in items]]
            if audit["task"] == "reach":
                anchor = support[sampler.integers(len(support))]
                neighbors = support[
                    np.linalg.norm(support - anchor, axis=1)
                    <= config.local_target_radius_m
                ]
                neighbor = neighbors[sampler.integers(len(neighbors))]
                requested.append(anchor + sampler.random() * (neighbor - anchor))
            else:
                requested.append(sampler.uniform(support.min(0), support.max(0)))
        requested = np.array(requested)
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
            state = fixed_observations(
                base,
                batch["positions"],
                batch["event_frames"],
                batch["hands"],
                skeleton,
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
            expanded_hands = batch["hands"].repeat_interleave(
                config.actions_per_condition
            )
            corrected = apply_fixed_residual(
                expanded_motion,
                actions,
                expanded_frames,
                expanded_hands,
                skeleton,
                config.max_angle,
                config.radius_frames,
            )
            joints = skeleton(corrected)
            metrics = fixed_metrics(
                joints, expanded_targets, expanded_hands, expanded_frames, audit["task"]
            )
            errors = metrics["error_m"]
            if audit["task"] == "reach":
                wrists = selected_wrists(joints, expanded_hands)
                window_frames = (
                    expanded_frames[:, None]
                    + torch.arange(-2, 3, device=config.device)[None]
                ).clamp(0, joints.shape[1] - 1)
                window_wrists = wrists[
                    torch.arange(len(joints), device=config.device)[:, None],
                    window_frames,
                ]
                objective = (
                    (window_wrists - expanded_targets[:, None])
                    .square()
                    .sum(-1)
                    .mean(-1)
                )
                reward = torch.exp(-objective / config.reward_scale_m**2)
            else:
                reward = torch.exp(-(errors / config.reward_scale_m).square()) * (
                    metrics["approach_speed_m_s"] / 0.8
                ).clamp(0, 1)
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
                deterministic = apply_fixed_residual(
                    base,
                    policy.mean(state),
                    batch["event_frames"],
                    batch["hands"],
                    skeleton,
                    config.max_angle,
                    config.radius_frames,
                )
                final_error = fixed_metrics(
                    skeleton(deterministic),
                    batch["positions"],
                    batch["hands"],
                    batch["event_frames"],
                    audit["task"],
                )["error_m"].mean()
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
            format="fixed_world_arm_ppo_v1",
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
