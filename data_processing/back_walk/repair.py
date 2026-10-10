"""Repair only existing back_walk rows; no reversal, synthetic motion or padding."""

from collections import Counter
import json
from pathlib import Path

import hydra
import numpy as np
from omegaconf import OmegaConf
from scipy.ndimage import uniform_filter1d
import torch

from data_processing.event_text import EventTextEncoder
from shared_motion.training.catalog import measure
from shared_motion.training.geometry import Skeleton
from shared_motion.training.model import file_sha256
from shared_motion.training.runner import save_json


def signals(joints):
    lateral = joints[:, 1, :2] - joints[:, 2, :2]
    lateral /= np.linalg.norm(lateral, axis=1, keepdims=True).clip(1e-8)
    forward = np.stack([lateral[:, 1], -lateral[:, 0]], axis=1)
    velocity = np.diff(joints[:, 0, :2], axis=0) * 20
    return (
        (velocity * forward[:-1]).sum(1),
        (velocity * lateral[:-1]).sum(1),
        np.unwrap(np.arctan2(lateral[:, 1], lateral[:, 0])),
    )


def backward_windows(joints, rules):
    forward, lateral, _ = signals(joints)
    smooth = uniform_filter1d(
        forward, size=rules.velocity_smoothing_frames, mode="nearest"
    )
    mask = smooth <= rules.maximum_forward_velocity_m_s
    changes = np.diff(np.pad(mask.astype(int), (1, 1)))
    windows = []
    for begin, end in zip(np.flatnonzero(changes == 1), np.flatnonzero(changes == -1)):
        active = np.flatnonzero(smooth[begin:end] < -rules.backward_edge_velocity_m_s)
        if not len(active):
            continue
        start = int(begin + active[0])
        stop = int(begin + active[-1] + 2)
        if stop - start < rules.minimum_frames:
            continue
        pieces = int(np.ceil((stop - start) / rules.maximum_frames))
        boundaries = np.linspace(start, stop, pieces + 1, dtype=int)
        windows.extend(
            (int(lower), int(upper))
            for lower, upper in zip(boundaries[:-1], boundaries[1:])
            if upper - lower >= rules.minimum_frames
        )
    return windows


def check_clip(joints, rules):
    forward, lateral, heading = signals(joints)
    total_path = np.abs(forward).sum()
    metrics = dict(
        mean_backward_speed_m_s=float(-forward.mean()),
        forward_path_fraction=float(
            np.maximum(forward, 0).sum() / max(total_path, 1e-8)
        ),
        lateral_path_ratio=float(np.abs(lateral).sum() / max(total_path, 1e-8)),
        heading_change_rad=float(np.ptp(heading)),
        left_ankle_height_range_m=float(np.ptp(joints[:, 7, 2])),
        right_ankle_height_range_m=float(np.ptp(joints[:, 8, 2])),
    )
    torso = (joints[:, 16] + joints[:, 17]) / 2 - joints[:, 0]
    metrics["maximum_torso_tilt_rad"] = float(
        np.arctan2(np.linalg.norm(torso[:, :2], axis=1), torso[:, 2]).max()
    )
    failures = []
    if metrics["maximum_torso_tilt_rad"] > rules.maximum_torso_tilt_rad:
        failures.append("bending_crouching_instead_of_upright_walking")
    if (
        not rules.minimum_mean_backward_speed_m_s
        <= metrics["mean_backward_speed_m_s"]
        <= rules.maximum_mean_backward_speed_m_s
    ):
        failures.append("no_sustained_backward_walk_speed")
    if metrics["forward_path_fraction"] > rules.maximum_forward_path_fraction:
        failures.append("forward_travel_remains")
    if metrics["lateral_path_ratio"] > rules.maximum_lateral_ratio:
        failures.append("sideways_or_diagonal_motion")
    if metrics["heading_change_rad"] > rules.maximum_heading_change_rad:
        failures.append("turning_motion")
    if (
        min(metrics["left_ankle_height_range_m"], metrics["right_ankle_height_range_m"])
        < rules.minimum_ankle_height_range_m
    ):
        failures.append("no_bilateral_stepping_evidence")
    return metrics, failures


@hydra.main(
    version_base="1.3", config_path="../../config", config_name="back_walk_repair"
)
def main(config):
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "cache").mkdir()
    torch.set_num_threads(2)
    skeleton = Skeleton(config.skeleton)
    encoder = EventTextEncoder(config.project, output)
    manifests = {
        split: json.loads((Path(config.base) / f"{split}.json").read_text())
        for split in ["train", "val"]
    }
    selected = [
        record
        for records in manifests.values()
        for record in records
        if record["task"] == "back_walk"
    ]
    accepted = []
    rejected = []
    before = []
    provenance = {}
    for record_index, record in enumerate(selected):
        with np.load(record["cache"]) as cache:
            real_motion = cache["motion"][: record["real_frames"]].copy()
        with torch.no_grad():
            old_joints = skeleton(torch.from_numpy(real_motion)[None])[0].numpy()
        old_metrics, _ = check_clip(old_joints, config.rules)
        before.append(
            dict(
                key=record["key"],
                split=record["split"],
                old_quantity=record["quantity"],
                old_pad_frames=record["pad_frames"],
                **old_metrics,
            )
        )
        full_source = (
            Path(config.project)
            / "outputs_amass/unified_direct_20261008/data/motions"
            / f'{record["family"]}.npy'
        )
        if full_source.exists():
            full = np.load(full_source, mmap_mode="r")
            source_start = max(
                0,
                int(
                    record.get(
                        "annotation_start_s", record["crop_start_frame_20fps"] / 20
                    )
                    * 20
                ),
            )
            source_stop = min(
                len(full),
                int(
                    record.get(
                        "annotation_end_frame_20fps",
                        record["crop_start_frame_20fps"] + record["real_frames"],
                    )
                ),
            )
            motion = np.asarray(full[source_start:source_stop]).copy()
            source = str(full_source)
            source_mode = "full_original_annotation"
        else:
            motion = real_motion
            source_start = record["crop_start_frame_20fps"]
            source = str(record["cache"])
            source_mode = "original_cache_real_frames_only"
        if source not in provenance:
            provenance[source] = file_sha256(source)
        if len(motion) < config.rules.minimum_frames:
            rejected.append(
                dict(
                    key=record["key"],
                    family=record["family"],
                    split=record["split"],
                    reasons=["too_short_without_padding"],
                )
            )
            continue
        with torch.no_grad():
            joints = skeleton(torch.from_numpy(motion)[None])[0].numpy()
        windows = backward_windows(joints, config.rules)
        admitted = 0
        for begin, end in windows:
            clip = motion[begin:end].copy()
            with torch.no_grad():
                crop_joints = skeleton(torch.from_numpy(clip)[None])[0].numpy()
                quantity = float(
                    measure(
                        skeleton,
                        torch.from_numpy(clip)[None],
                        torch.tensor([6]),
                        torch.tensor([len(clip)]),
                    )[0]
                )
            metrics, failures = check_clip(crop_joints, config.rules)
            if quantity <= 0:
                failures.append("nonpositive_actual_task_quantity")
            if failures:
                rejected.append(
                    dict(
                        key=record["key"],
                        family=record["family"],
                        split=record["split"],
                        start=source_start + begin,
                        stop=source_start + end,
                        reasons=failures,
                        metrics=metrics,
                    )
                )
                continue
            start = source_start + begin
            stop = source_start + end
            key = f'back_walk_{record["key"]}_{start}_{stop}'
            cache_path = output / "cache" / f'{record["split"]}_{key}.npz'
            caption = "walk backwards"
            global_text, local_text, local_mask = encoder.make_arrays(
                caption,
                np.arange(start, stop),
                start / 20,
                stop / 20,
                ["action", "left_leg", "right_leg"],
            )
            np.savez(
                cache_path,
                motion=clip,
                tx=global_text,
                local=local_text,
                local_mask=local_mask,
                quantity=np.float32(quantity),
                task=np.int64(6),
            )
            accepted.append(
                dict(
                    split=record["split"],
                    task="back_walk",
                    task_id=6,
                    key=key,
                    family=record["family"],
                    caption=caption,
                    original_caption=record["caption"],
                    original_key=record["key"],
                    original_cache=record["cache"],
                    original_quantity=record["quantity"],
                    original_crop_start_20fps=record["crop_start_frame_20fps"],
                    original_real_frames=record["real_frames"],
                    annotation_start_s=record.get("annotation_start_s"),
                    annotation_end_frame_20fps=record.get("annotation_end_frame_20fps"),
                    raw_path=record.get("raw_path"),
                    motion_source=source,
                    source_mode=source_mode,
                    source_slice_start=begin
                    + (
                        source_start if source_mode == "full_original_annotation" else 0
                    ),
                    source_slice_stop=end
                    + (
                        source_start if source_mode == "full_original_annotation" else 0
                    ),
                    crop_start_frame_20fps=start,
                    crop_end_frame_20fps=stop,
                    real_frames=len(clip),
                    target_frames=len(clip),
                    pad_frames=0,
                    quantity=quantity,
                    cache=str(cache_path),
                    cache_sha256=file_sha256(cache_path),
                    source_kind="existing_back_walk_source_direction_verified_crop",
                    annotation_level="kinematic_crop_of_existing_annotation",
                    caption_basis="Canonical crop descriptor after direction and bilateral-step checks; original caption retained separately",
                    metrics=metrics,
                    ready_for_training=True,
                    manual_verified=False,
                    review_status="awaiting_user_review",
                )
            )
            admitted += 1
        if not admitted:
            rejected.append(
                dict(
                    key=record["key"],
                    family=record["family"],
                    split=record["split"],
                    reasons=["no_admissible_continuous_backward_segment"],
                    original_metrics=old_metrics,
                )
            )
        if (record_index + 1) % 200 == 0:
            print(
                json.dumps(
                    dict(
                        processed=record_index + 1,
                        total=len(selected),
                        accepted=len(accepted),
                    )
                ),
                flush=True,
            )
    encoder.save()
    for split, original in manifests.items():
        save_json(
            output / f"{split}.json",
            [record for record in original if record["task"] != "back_walk"]
            + [record for record in accepted if record["split"] == split],
        )
        provenance[str(Path(config.base) / f"{split}.json")] = file_sha256(
            Path(config.base) / f"{split}.json"
        )
    for path in [
        Path(__file__),
        Path(config.skeleton),
        Path(__file__).parents[1] / "event_text.py",
    ]:
        provenance[str(path)] = file_sha256(path)
    save_json(output / "back_walk_index.json", accepted)
    save_json(output / "rejected.json", rejected)
    save_json(output / "before.json", before)
    save_json(output / "provenance.json", provenance)
    OmegaConf.save(config, output / "config.yaml", resolve=True)
    audit = dict(
        base=config.base,
        original_counts={
            split: sum(record["split"] == split for record in selected)
            for split in ["train", "val"]
        },
        counts={
            split: sum(record["split"] == split for record in accepted)
            for split in ["train", "val"]
        },
        families={
            split: len(
                {record["family"] for record in accepted if record["split"] == split}
            )
            for split in ["train", "val"]
        },
        original_nonpositive_labels=sum(
            record["old_quantity"] <= 0 for record in before
        ),
        original_forward_contaminated=sum(
            record["forward_path_fraction"] > config.rules.maximum_forward_path_fraction
            for record in before
        ),
        accepted_original_records=len(
            {(record["split"], record["original_key"]) for record in accepted}
        ),
        original_records_rejected=len(selected)
        - len({(record["split"], record["original_key"]) for record in accepted}),
        maximum_forward_path_fraction=max(
            record["metrics"]["forward_path_fraction"] for record in accepted
        ),
        other19_tasks_unchanged=True,
        training_started=False,
        limitations=[
            "Direction and bilateral ankle motion verify cropping, not complete semantic purity.",
            "Canonical backward caption replaces whole-sequence descriptions that may mention forward phases; originals remain auditable.",
            "No new source families, mirroring, time reversal, amplitude scaling or padding.",
        ],
    )
    save_json(output / "audit.json", audit)
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
