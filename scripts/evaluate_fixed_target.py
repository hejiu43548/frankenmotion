"""Paired fixed-world target evaluation for reach and strike."""

import itertools
import json
from pathlib import Path
import sys
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
)


@hydra.main(
    version_base="1.3", config_path="../config", config_name="evaluate_fixed_target"
)
def main(config):
    torch.set_num_threads(4)
    checkpoint = torch.load(config.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint.get("format") != "fixed_world_adapter_v1":
        raise ValueError("Expected fixed-world adapter")
    training = OmegaConf.create(checkpoint["config"])
    audit = checkpoint["audit"]
    rows, items = load_items(training.data, config.split)
    model = (
        ReachDiffusion(
            instantiate(training.backbone),
            audit["train_target_center"],
            audit["train_target_scale"],
        )
        .to(config.device)
        .eval()
    )
    model.load_adapter(checkpoint["adapter"])
    skeleton = Skeleton(training.skeleton).to(config.device)
    policy = None
    if config.policy:
        policy_checkpoint = torch.load(
            config.policy, map_location="cpu", weights_only=False
        )
        if policy_checkpoint.get(
            "format"
        ) != "fixed_world_arm_ppo_v1" or policy_checkpoint[
            "initial_sha256"
        ] != file_sha256(
            config.checkpoint
        ):
            raise ValueError("Policy must bind to exact fixed-world generator")
        policy = FixedTargetPolicy().to(config.device).eval()
        policy.load_state_dict(policy_checkpoint["policy"], strict=True)
    requests = [
        (index, row["key"], row["target_xyz_m"]) for index, row in enumerate(rows)
    ]
    if config.target_mode == "grid":
        if config.split != "train":
            raise ValueError("Construct interpolation grid from train only")
        requests = []
        for template_index in config.template_indices:
            hand = int(items[template_index]["hands"])
            support = np.array(
                [
                    row["target_xyz_m"]
                    for row, item in zip(rows, items)
                    if int(item["hands"]) == hand
                ]
            )
            for grid_index, fractions in enumerate(
                itertools.product(config.grid_fractions, repeat=3)
            ):
                target = support.min(0) + np.array(fractions) * (
                    support.max(0) - support.min(0)
                )
                if (
                    np.linalg.norm(support - target, axis=1).min()
                    >= config.min_train_distance_m
                ):
                    requests.append(
                        (
                            template_index,
                            f"template{template_index}_hole_{grid_index:02d}",
                            target.tolist(),
                        )
                    )
    elif config.target_mode != "source":
        raise ValueError("Unknown target_mode")
    output = Path(config.output)
    if (output / "results.json").exists():
        raise ValueError("Refusing to overwrite evaluation")
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output / "config.yaml", resolve=True)
    records = []
    for item_index, key, world_target in requests:
        batch = make_batch(items, [item_index] * len(config.seeds), config.device)
        # This point is established before all three generations and never recomputed.
        targets = torch.tensor(world_target, device=config.device, dtype=torch.float32)[
            None
        ].repeat(len(config.seeds), 1)
        with torch.no_grad():
            for mode in (
                ["official", "adapter", "ppo"]
                if policy is not None
                else ["official", "adapter"]
            ):
                if mode != "ppo":
                    generated = model.sample(
                        batch["tx"],
                        batch["local"],
                        batch["local_mask"],
                        batch["mask"],
                        batch["hands"],
                        targets,
                        config.seeds,
                        steps=config.steps,
                        enabled=mode != "official",
                    )
                else:
                    state = fixed_observations(
                        generated,
                        targets,
                        batch["event_frames"],
                        batch["hands"],
                        skeleton,
                    )
                    generated = apply_fixed_residual(
                        generated,
                        policy.mean(state),
                        batch["event_frames"],
                        batch["hands"],
                        skeleton,
                        policy_checkpoint["config"]["max_angle"],
                        policy_checkpoint["config"]["radius_frames"],
                    )
                joints = skeleton(generated)
                metrics = fixed_metrics(
                    joints,
                    targets,
                    batch["hands"],
                    batch["event_frames"],
                    audit["task"],
                )
                np.savez(
                    output / f"{key}_{mode}.npz",
                    motion=generated.cpu().numpy(),
                    joints=joints.cpu().numpy(),
                    target=targets.cpu().numpy(),
                    target_world=targets.cpu().numpy(),
                    hands=batch["hands"].cpu().numpy(),
                    event_frames=batch["event_frames"].cpu().numpy(),
                    seeds=np.array(config.seeds),
                )
                for sample_index, seed in enumerate(config.seeds):
                    records.append(
                        dict(
                            key=key,
                            mode=mode,
                            seed=int(seed),
                            hand=int(batch["hands"][sample_index]),
                            target_world_m=world_target,
                            **{
                                name: value[sample_index].item()
                                for name, value in metrics.items()
                            },
                        )
                    )
    if not records:
        raise ValueError("Evaluation has no requests")
    summary = {}
    for mode in sorted({record["mode"] for record in records}):
        selected = [record for record in records if record["mode"] == mode]
        summary[mode] = dict(
            samples=len(selected),
            mean_error_m=float(np.mean([record["error_m"] for record in selected])),
            within_10cm=float(
                np.mean([record["error_m"] < 0.1 for record in selected])
            ),
            strict_pass=float(np.mean([record["strict_pass"] for record in selected])),
        )
    result = dict(
        target_frame="fixed_world",
        task=audit["task"],
        requests=len(requests),
        checkpoint_sha256=file_sha256(config.checkpoint),
        policy_sha256=file_sha256(config.policy) if config.policy else None,
        summary=summary,
        records=records,
        note="Target/noise tests, not independent real source motions; no physical contact simulation. All modes share the exact externally specified world point.",
    )
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
