"""Relabel audited real-motion caches with world-fixed wrist endpoints.

Input is the verified cache protocol used by train_strike3d. Reach imports must
first supply reviewed hand and event frames; this script never invents those.
"""

import json
from pathlib import Path
import sys
import hydra
import numpy as np
from omegaconf import OmegaConf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_strike3d import load_items
from shared_motion.training.geometry import Skeleton
from shared_motion.training.fixed_target import body_basis
from shared_motion.training.model import file_sha256


@hydra.main(
    version_base="1.3", config_path="../config", config_name="prepare_fixed_targets"
)
def main(config):
    torch.set_num_threads(4)
    if config.task not in ("reach", "strike"):
        raise ValueError("Unknown task")
    output = Path(config.output)
    if (output / "audit.json").exists():
        raise ValueError("Refusing to overwrite prepared data")
    (output / "cache").mkdir(parents=True, exist_ok=True)
    skeleton = Skeleton(config.skeleton)
    manifests = {}
    for split in ["train", "val"]:
        rows, items = load_items(config.input, split)
        converted = []
        for row, item in zip(rows, items):
            joints = skeleton(item["motion"][None])[0, :, :22]
            hand = int(item["hands"])
            event = int(item["event_frames"])
            if hand not in (0, 1) or not 0 <= event < len(joints):
                raise ValueError("Reviewed hand/event is missing or invalid")
            target_world = joints[event, 20 + hand]
            initial_origin = joints[0, 0]
            initial_basis = body_basis(joints[0])
            relative = initial_basis.T @ (target_world - initial_origin)
            cache = output / "cache" / (row["key"] + ".npz")
            arrays = {name: value.numpy() for name, value in item.items()}
            arrays["positions"] = target_world.numpy()
            np.savez(cache, **arrays)
            converted.append(
                dict(
                    row,
                    cache=str(cache.resolve()),
                    rebuilt_cache_sha256=file_sha256(cache),
                    source_prepared_cache_sha256=row["rebuilt_cache_sha256"],
                    target_frame="fixed_world",
                    task=config.task,
                    hand=hand,
                    target_xyz_m=target_world.tolist(),
                    initial_body_target_xyz_m=relative.tolist(),
                    initial_pelvis_world_m=initial_origin.tolist(),
                    initial_body_basis=initial_basis.tolist(),
                )
            )
        manifests[split] = converted
        (output / f"{split}.json").write_text(json.dumps(converted, indent=2) + "\n")
    if {row["family"] for row in manifests["train"]} & {
        row["family"] for row in manifests["val"]
    }:
        raise ValueError("Source family leakage")
    train_targets = np.array([row["target_xyz_m"] for row in manifests["train"]])
    original = json.loads((Path(config.input) / "audit.json").read_text())
    audit = dict(
        target_frame="fixed_world",
        task=config.task,
        counts={split: len(rows) for split, rows in manifests.items()},
        hands=sorted({row["hand"] for row in manifests["train"]}),
        train_target_center=train_targets.mean(0).tolist(),
        train_target_scale=np.maximum(train_targets.std(0), 0.05).tolist(),
        train_target_min=train_targets.min(0).tolist(),
        train_target_max=train_targets.max(0).tolist(),
        source_audit=original,
        source_manifest_hashes={
            split: file_sha256(Path(config.input) / f"{split}.json")
            for split in manifests
        },
        skeleton_sha256=file_sha256(config.skeleton),
        definition="Single wrist world point at the reviewed source event. Initial-frame relative command is converted once using the supplied initial pose. No per-frame target update.",
    )
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    OmegaConf.save(config, output / "config.yaml", resolve=True)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
