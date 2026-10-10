"""Matched text/noise official-vs-adapter endpoint diagnostics, without teachers."""

import json
import itertools
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
from shared_motion.training.reach3d_geometry import wrist_positions_in_body_frame
from shared_motion.training.reach3d_model import ReachDiffusion


@hydra.main(
    version_base="1.3", config_path="../config", config_name="evaluate_strike3d"
)
def main(config):
    torch.set_num_threads(4)
    checkpoint = torch.load(config.checkpoint, map_location="cpu", weights_only=False)
    training = OmegaConf.create(checkpoint["config"])
    rows, items = load_items(training.data, config.split)
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
    skeleton = Skeleton(training.skeleton).to(config.device)
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    records = []
    requests = [
        (index, row["key"], row["target_xyz_m"]) for index, row in enumerate(rows)
    ]
    if config.target_mode == "holes":
        if config.split != "train":
            raise ValueError("Hole grid must be constructed from training support only")
        targets = np.array([row["target_xyz_m"] for row in rows])
        lower, upper = targets.min(0), targets.max(0)
        requests = []
        for grid_index, fractions in enumerate(
            itertools.product([0.2, 0.5, 0.8], repeat=3)
        ):
            target = lower + np.asarray(fractions) * (upper - lower)
            if np.linalg.norm(targets - target, axis=1).min() >= 0.04:
                requests.append((0, f"hole_{grid_index:02d}", target.tolist()))
    elif config.target_mode != "source":
        raise ValueError("Unknown target mode")
    for index, request_key, target in requests:
        batch = make_batch(items, [index] * len(config.seeds), config.device)
        batch["positions"] = torch.tensor(
            target, device=config.device, dtype=torch.float32
        )[None].repeat(len(config.seeds), 1)
        for enabled in [False, True]:
            generated = model.sample(
                batch["tx"],
                batch["local"],
                batch["local_mask"],
                batch["mask"],
                batch["hands"],
                batch["positions"],
                seeds=config.seeds,
                steps=config.steps,
                enabled=enabled,
            )
            with torch.no_grad():
                joints = skeleton(generated)
                wrists = wrist_positions_in_body_frame(joints[..., :22, :])
                measured = wrists[
                    torch.arange(len(generated), device=config.device),
                    batch["event_frames"],
                    1,
                ]
                errors = (measured - batch["positions"]).norm(dim=-1)
                root_excursion = (
                    (joints[:, :, 0] - joints[:, :1, 0]).norm(dim=-1).max(dim=-1).values
                )
                wrist_speed = (
                    ((joints[:, 1:, 21] - joints[:, :-1, 21]) * 20)
                    .norm(dim=-1)
                    .max(dim=-1)
                    .values
                )
            mode = "adapter" if enabled else "official"
            np.savez(
                output / f"{request_key}_{mode}.npz",
                motion=generated.cpu().numpy(),
                joints=joints.cpu().numpy(),
                target=batch["positions"].cpu().numpy(),
                event_frames=batch["event_frames"].cpu().numpy(),
                seeds=np.array(config.seeds),
            )
            for sample_index, seed in enumerate(config.seeds):
                records.append(
                    dict(
                        key=request_key,
                        mode=mode,
                        seed=seed,
                        error_m=float(errors[sample_index]),
                        target=batch["positions"][sample_index].tolist(),
                        measured=measured[sample_index].tolist(),
                        root_excursion_m=float(root_excursion[sample_index]),
                        peak_wrist_speed_m_s=float(wrist_speed[sample_index]),
                    )
                )
    summary = {
        mode: dict(
            mean_error_m=float(
                np.mean([row["error_m"] for row in records if row["mode"] == mode])
            ),
            within_10cm=float(
                np.mean(
                    [row["error_m"] < 0.1 for row in records if row["mode"] == mode]
                )
            ),
        )
        for mode in ["official", "adapter"]
    }
    result = dict(
        split=config.split,
        samples=len(requests),
        target_mode=config.target_mode,
        seeds=list(config.seeds),
        summary=summary,
        records=records,
        note="Geometric endpoint diagnostic. Not contact success or independent generalization proof.",
    )
    (output / "results.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
