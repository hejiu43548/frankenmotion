"""Inventory scene01-01 and locate candidate leftward non-crossing runs."""

import json
from pathlib import Path

import hydra
import numpy as np
from scipy.ndimage import binary_closing
from scipy.ndimage import uniform_filter1d
import torch

from scripts.prepare_amass20 import convert_source
from shared_motion.training.geometry import Skeleton
from shared_motion.training.model import file_sha256
from shared_motion.training.runner import save_json


def movement_signals(joints):
    lateral = joints[:, 1, :2] - joints[:, 2, :2]
    lateral /= np.linalg.norm(lateral, axis=1, keepdims=True).clip(1e-8)
    forward = np.stack([lateral[:, 1], -lateral[:, 0]], axis=1)
    velocity = np.gradient(joints[:, 0, :2], axis=0) * 20
    return dict(
        lateral_velocity=(velocity * lateral).sum(1),
        forward_velocity=(velocity * forward).sum(1),
        ankle_separation=((joints[:, 7, :2] - joints[:, 8, :2]) * lateral).sum(1),
        heading=np.unwrap(np.arctan2(lateral[:, 1], lateral[:, 0])),
    )


def runs(mask):
    changes = np.diff(np.pad(mask.astype(int), (1, 1)))
    return list(
        zip(
            np.flatnonzero(changes == 1).tolist(),
            np.flatnonzero(changes == -1).tolist(),
        )
    )


@hydra.main(
    version_base="1.3", config_path="../../config", config_name="sidestep_repair"
)
def main(config):
    output = Path(config.output) / "inspection"
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    annotation_root = (
        Path(config.project) / "datasets/annotations/frankenstein-dataset/annotations"
    )
    annotations = json.loads((annotation_root / "annotations.json").read_text())
    splits = {}
    for split in ["train", "val", "test"]:
        for key in (annotation_root / "splits" / f"{split}.txt").read_text().split():
            family = annotations[key]["path"]
            assert splits.setdefault(family, split) == split
    skeleton = Skeleton(config.skeleton)
    records = []
    for raw in sorted((Path(config.amass) / "MPI_HDM05").glob("*/*01-01*.npz")):
        family = str(raw.relative_to(config.amass).with_suffix(""))
        split = splits.get(family)
        if split not in ["train", "val"]:
            records.append(
                dict(family=family, split=split, status="held_out_no_motion_inspection")
            )
            continue
        motion_path = (
            Path(config.project)
            / "outputs_amass/unified_direct_20261008/data/motions"
            / f"{family}.npy"
        )
        if not motion_path.exists():
            motion_path = output / "motions" / f"{family}.npy"
            result = convert_source(
                (
                    str(Path(config.project) / "work/unified_direct/prepare_data.py"),
                    (family, str(raw), str(motion_path), config.skeleton),
                )
            )
            assert result["status"] in ["ok", "cached"], result
        motion = np.load(motion_path)
        with torch.no_grad():
            joints = skeleton(torch.from_numpy(motion)[None])[0].numpy()
        signals = movement_signals(joints)
        lateral = uniform_filter1d(
            signals["lateral_velocity"], size=config.rules.velocity_smoothing_frames
        )
        forward = uniform_filter1d(
            signals["forward_velocity"], size=config.rules.velocity_smoothing_frames
        )
        active = (
            (lateral > config.rules.minimum_lateral_speed_m_s)
            & (
                np.abs(forward)
                < np.maximum(0.1, config.rules.maximum_forward_speed_ratio * lateral)
            )
            & (signals["ankle_separation"] > config.rules.minimum_ankle_separation_m)
        )
        active = binary_closing(
            active, structure=np.ones(config.rules.maximum_gap_frames)
        )
        candidates = []
        for start, stop in runs(active):
            displacement = float(np.sum(signals["lateral_velocity"][start:stop]) / 20)
            if (
                stop - start < 20
                or displacement < config.rules.minimum_event_displacement_m
            ):
                continue
            candidates.append(
                dict(
                    start=start,
                    stop=stop,
                    start_s=start / 20,
                    end_s=stop / 20,
                    left_distance_m=displacement,
                    heading_range_rad=float(np.ptp(signals["heading"][start:stop])),
                    ankle_min_m=float(signals["ankle_separation"][start:stop].min()),
                )
            )
        np.savez_compressed(output / f"{raw.stem}.npz", joints=joints, **signals)
        with np.load(raw) as raw_archive:
            original_frames = len(raw_archive["poses"])
            original_fps = float(raw_archive["mocap_framerate"])
        record = dict(
            family=family,
            split=split,
            status="inspected",
            raw=str(raw),
            raw_sha256=file_sha256(raw),
            motion=str(motion_path),
            motion_sha256=file_sha256(motion_path),
            frames=len(motion),
            raw_frames=original_frames,
            raw_fps=original_fps,
            candidates=candidates,
        )
        records.append(record)
        print(
            json.dumps(dict(family=family, split=split, candidates=candidates)),
            flush=True,
        )
    save_json(output / "inventory.json", records)


if __name__ == "__main__":
    main()
