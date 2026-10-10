"""Encode rule-rebuilt real reach crops for the standalone fixed-target adapter.

Selection is produced by point.repair inspect_only using the committed thresholds.
This importer preserves source frames, requires aggregate parity with the published
review, and records that original Betail cache byte parity is not established.
"""

import json
from pathlib import Path
import sys
import clip
import hydra
import numpy as np
from omegaconf import OmegaConf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared_motion.training.geometry import Skeleton
from shared_motion.training.model import file_sha256
from shared_motion.training.fixed_target import body_basis


@hydra.main(
    version_base="1.3",
    config_path="../../config",
    config_name="prepare_reach_fixed_training",
)
def main(config):
    torch.set_num_threads(4)
    output = Path(config.output)
    if (output / "audit.json").exists():
        raise ValueError("Prepared output already exists")
    (output / "cache").mkdir(parents=True, exist_ok=True)
    source_rows = json.loads(Path(config.index).read_text())
    published = json.loads(Path(config.review_state).read_text())["tasks"]["reach"][
        "coverage"
    ]
    for split in ["train", "val"]:
        rows = [row for row in source_rows if row["split"] == split]
        targets = np.array([row["target_xyz_m"] for row in rows])
        expected = published[split]
        if (
            len(rows) != expected["samples"]
            or len({row["family"] for row in rows}) != expected["families"]
        ):
            raise ValueError("Reconstruction does not match published source counts")
        if not np.allclose(
            targets.min(0), expected["xyz_min_m"], atol=1e-6
        ) or not np.allclose(targets.max(0), expected["xyz_max_m"], atol=1e-6):
            raise ValueError("Reconstruction does not match published XYZ range")
    if {row["family"] for row in source_rows if row["split"] == "train"} & {
        row["family"] for row in source_rows if row["split"] == "val"
    }:
        raise ValueError("Source-family leakage")
    encoder, _ = clip.load(config.clip_checkpoint, device="cpu")
    encoder.eval()
    labels = sorted({row["caption"] for row in source_rows})
    with torch.no_grad():
        encoded = encoder.encode_text(clip.tokenize(labels)).float().numpy()
    embeddings = dict(zip(labels, encoded))
    projection = np.load(config.pca)
    parts = json.loads(Path(config.official_config).read_text())["data"][
        "text_encoder"
    ]["body_part_order"]
    skeleton = Skeleton(config.skeleton)
    converted = []
    for row in source_rows:
        if row["semantic_revision"] != "right_wrist_xyz_v1" or row["pad_frames"] != 0:
            raise ValueError("Expected cleaned, unpadded right-wrist reach XYZ data")
        motion = np.load(row["motion_source"], mmap_mode="r")[
            row["crop_start_frame_20fps"] : row["crop_end_frame_20fps"]
        ].copy()
        joints = skeleton(torch.from_numpy(motion).float()[None])[0]
        world_target = joints[-5:, 21].mean(0)
        relative = body_basis(joints[0]).T @ (world_target - joints[0, 0])
        if not torch.allclose(relative, torch.tensor(row["target_xyz_m"]), atol=1e-6):
            raise ValueError("Recomputed final-hold target mismatch")
        text = embeddings[row["caption"]]
        reduced = (text - projection["mean"]) @ projection["components"].T
        local = np.zeros((len(motion), 408), np.float32)
        local_mask = np.zeros_like(local, dtype=bool)
        for part_index, part in enumerate(parts):
            if part in ["action", "right_arm"]:
                local[:, part_index * 51 : (part_index + 1) * 51] = reduced
                local_mask[:, part_index * 51 : (part_index + 1) * 51] = True
        cache = output / "cache" / (row["key"] + ".npz")
        np.savez(
            cache,
            motion=motion,
            local=local,
            local_mask=local_mask,
            tx=text,
            hands=np.int64(1),
            positions=world_target.numpy(),
            event_frames=np.int64(len(motion) - 3),
        )
        converted.append(
            dict(
                row,
                source_initial_body_target_xyz_m=row["target_xyz_m"],
                target_xyz_m=world_target.tolist(),
                target_frame="fixed_world",
                hand=1,
                event_frame=len(motion) - 3,
                initial_pelvis_world_m=joints[0, 0].tolist(),
                cache=str(cache.resolve()),
                rebuilt_cache_sha256=file_sha256(cache),
                source_motion_sha256=file_sha256(row["motion_source"]),
            )
        )
    for split in ["train", "val"]:
        (output / f"{split}.json").write_text(
            json.dumps([row for row in converted if row["split"] == split], indent=2)
            + "\n"
        )
    train_targets = np.array(
        [row["target_xyz_m"] for row in converted if row["split"] == "train"]
    )
    audit = dict(
        task="reach",
        target_frame="fixed_world",
        hands=[1],
        counts={
            split: sum(row["split"] == split for row in converted)
            for split in ["train", "val"]
        },
        train_target_center=train_targets.mean(0).tolist(),
        train_target_scale=np.maximum(train_targets.std(0), 0.05).tolist(),
        train_target_min=train_targets.min(0).tolist(),
        train_target_max=train_targets.max(0).tolist(),
        source_index_sha256=file_sha256(config.index),
        source_review_sha256=file_sha256(config.review_state),
        pca_sha256=file_sha256(config.pca),
        skeleton_sha256=file_sha256(config.skeleton),
        original_cache_byte_parity=False,
        selection="Committed colleague rules applied to locally available real AMASS/BABEL. Counts, families, XYZ extrema match published review; original Betail per-row cache manifests were not accessible.",
        target_definition="Mean right wrist WORLD position over final 5 valid frames; fixed before generation. Midpoint event index is frames-3. Not moving-body relative.",
    )
    (output / "audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    first = next(row for row in converted if row["split"] == "train")
    with np.load(first["cache"]) as archive:
        np.savez(
            output / "text_template.npz",
            **{
                name: archive[name]
                for name in ["tx", "local", "local_mask", "event_frames"]
            },
        )
    OmegaConf.save(config, output / "config.yaml", resolve=True)
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
