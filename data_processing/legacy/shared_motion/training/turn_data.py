"""Verify and reuse the reviewed event sources; never import single-task models."""

from collections import Counter
import json
from pathlib import Path

import numpy as np
from omegaconf import OmegaConf
import torch

from .geometry import Skeleton
from .model import file_sha256
from .turn import TURN_NATIVE_SPEED, TURN_POLICY, TURN_REVISION, signed_turn
from .turn import validate_turn_record


def read_json(path):
    return json.loads(Path(path).read_text())


def check_hashes(directory, expected):
    for name, digest in expected.items():
        if file_sha256(directory / name) != digest:
            raise ValueError(f"Source provenance mismatch: {directory / name}")


def check_admission(row):
    """Recheck the recorded v2 criteria, not a caption-only turn filter."""
    metrics = row["metrics"]
    angle = metrics["body_turn_rad"]
    path_angle = metrics["path_turn_rad"]
    conditions = [
        row["admission_reasons"] == [],
        row["pad_frames"] == 0,
        40 <= row["real_frames"] <= 120,
        metrics["walking_annotation_fraction"] >= 0.85,
        metrics["conflicting_action_fraction"] == 0,
        metrics["path_m"] >= 0.5,
        metrics["excursion_m"] >= 0.3,
        0.2 <= metrics["mean_speed_m_s"] <= 1.8,
        metrics["event_path_m"] >= 0.3,
        metrics["event_excursion_m"] >= 0.2,
        metrics["moving_during_turn_fraction"] >= 0.7,
        metrics["yaw_weighted_moving_fraction"] >= 0.8,
        0.35 <= abs(angle) <= 3.5,
        metrics["direction_consistency"] >= 0.7,
        metrics["forward_alignment_fraction"] >= 0.7,
        min(metrics["entry_speed_m_s"], metrics["exit_speed_m_s"]) >= 0.15,
        abs(path_angle) >= 0.25,
        angle * path_angle > 0,
        abs(angle - path_angle) <= 0.6,
        min(metrics["ankle_relative_excursion_m"]) >= 0.08,
        metrics["root_height_range_m"] <= 0.2,
    ]
    if not all(conditions):
        raise ValueError(f"Event does not pass walking-turn v2 admission: {row['id']}")


def replace_turn_rows(original, replacement, split):
    if any(row["split"] != split for row in original + replacement):
        raise ValueError("Unexpected split in source manifest")
    if any(row["task"] != "turn" for row in replacement):
        raise ValueError("Replacement must contain turn events only")
    if len({row["id"] for row in replacement}) != len(replacement):
        raise ValueError("Duplicate turn event IDs")
    # Keep all non-turn dictionaries, ordering and relative cache paths verbatim.
    return [row for row in original if row["task"] != "turn"] + replacement


def repair(config):
    output = Path(config.output).resolve()
    if output.exists():
        raise FileExistsError(f"Choose a fresh output directory: {output}")
    admission = Path(config.admission_dir).resolve()
    prepared = Path(config.prepared_dir).resolve()
    if config.source.revision != TURN_REVISION:
        raise ValueError("Expected the reviewed walking_turn_v2 source revision")
    check_hashes(admission, config.source.admission_sha256)
    check_hashes(prepared, config.source.prepared_sha256)
    provenance = read_json(admission / "provenance.json")
    if file_sha256(config.skeleton) != provenance["skeleton_sha256"]:
        raise ValueError("Skeleton differs from walking-turn source selection")
    rules = read_json(admission / "rules.json")
    if rules["version"] != TURN_REVISION:
        raise ValueError("Unexpected turn admission rules")
    skeleton = Skeleton(config.skeleton)
    torch.set_num_threads(2)
    motion_hashes = read_json(prepared / "source_sha256.json")
    checked_sources = set()
    outputs = {}
    by_split = {}
    speeds = []
    turn_index = []
    maximum_label_error = 0.0
    held_out = read_json(admission / "turn_test.json")
    test_families = {row["family"] for row in held_out}
    for split in ["train", "val"]:
        original_path = Path(config[f"base_{split}_manifest"])
        original = read_json(original_path)
        selected = read_json(admission / f"turn_{split}.json")
        candidates = [
            row
            for row in read_json(prepared / f"{split}.json")
            if row["task"] == "turn"
        ]
        by_id = {row["id"]: row for row in candidates}
        if len(by_id) != len(candidates) or set(by_id) != {
            row["id"] for row in selected
        }:
            raise ValueError("Prepared turn events do not match the admitted set")
        replacement = []
        for event in selected:
            check_admission(event)
            row = dict(by_id[event["id"]])
            for field in [
                "family",
                "split",
                "direction",
                "crop_start_frame_20fps",
                "crop_end_frame_20fps",
                "real_frames",
                "metrics",
            ]:
                if row[field] != event[field]:
                    raise ValueError(f"Changed admitted event field: {field}")
            if row["annotation_key"] != event["key"]:
                raise ValueError("Event uses a different text annotation")
            cache = Path(row["cache"])
            digest = file_sha256(cache)
            if digest != row["cache_sha256"]:
                raise ValueError(f"Changed regenerated cache: {cache}")
            source = Path(row["motion_source"])
            if str(source) not in checked_sources:
                if file_sha256(source) != motion_hashes[str(source)]:
                    raise ValueError(f"Changed full motion source: {source}")
                checked_sources.add(str(source))
            with np.load(cache, allow_pickle=False) as archive:
                motion = archive["motion"].copy()
                quantity = float(archive["quantity"])
                frames = len(motion)
                if (
                    archive["local"].shape != (frames, 408)
                    or archive["local_mask"].shape != (frames, 408)
                    or archive["tx"].shape != (512,)
                    or not all(
                        np.isfinite(archive[name]).all()
                        for name in ["motion", "local", "tx"]
                    )
                ):
                    raise ValueError("Invalid regenerated text/motion cache")
            start, stop = row["crop_start_frame_20fps"], row["crop_end_frame_20fps"]
            full_motion = np.load(source, mmap_mode="r")
            if not np.array_equal(motion, full_motion[start:stop]):
                raise ValueError("Cache is not the admitted event's exact motion crop")
            with torch.no_grad():
                measured = float(
                    signed_turn(skeleton(torch.from_numpy(motion)[None]))[0]
                )
            error = abs(measured - quantity)
            if error > 1e-4 or abs(quantity - row["quantity"]) > 1e-6:
                raise ValueError("Turn label does not match the unwrapped actual crop")
            maximum_label_error = max(maximum_label_error, error)
            row["turn_source_revision"] = TURN_REVISION
            row["turn_admission_sha256"] = config.source.admission_sha256[
                f"turn_{split}.json"
            ]
            row["turn_pca_sha256"] = config.source.prepared_sha256["pca.npz"]
            validate_turn_record(row, digest, frames, quantity)
            replacement.append(row)
            if split == "train":
                speeds.append(
                    float(np.linalg.norm(motion[:-1, 1:3], axis=-1).mean() * 20)
                )
            turn_index.append(
                {
                    name: row[name]
                    for name in [
                        "id",
                        "split",
                        "family",
                        "annotation_key",
                        "cache",
                        "cache_sha256",
                        "crop_start_frame_20fps",
                        "crop_end_frame_20fps",
                        "direction",
                        "quantity",
                    ]
                }
            )
        outputs[split] = replace_turn_rows(original, replacement, split)
        if {row["family"] for row in outputs[split]} & test_families:
            raise ValueError("Held-out test turn family leaked into train/val")
        by_split[split] = dict(
            original_rows=len(original),
            removed_legacy_turn=sum(row["task"] == "turn" for row in original),
            admitted_turn=len(replacement),
            unchanged_other_rows=len(original)
            - sum(row["task"] == "turn" for row in original),
            total_rows=len(outputs[split]),
            task_counts=dict(Counter(row["task"] for row in outputs[split])),
            directions=dict(Counter(row["direction"] for row in replacement)),
            base_manifest_sha256=file_sha256(original_path),
        )
    if {row["family"] for row in outputs["train"]} & {
        row["family"] for row in outputs["val"]
    }:
        raise ValueError("Source family leakage after turn replacement")
    if abs(float(np.median(speeds)) - TURN_NATIVE_SPEED) > 1e-7:
        raise ValueError("Fixed turn speed differs from the TRAIN-only median")
    output.mkdir(parents=True)
    payloads = dict(
        outputs,
        turn_index=turn_index,
        turn_policy=TURN_POLICY,
        admission_rules=rules,
        audit=dict(
            revision=TURN_REVISION,
            by_split=by_split,
            verified_turn_caches=len(turn_index),
            verified_full_motion_sources=len(checked_sources),
            max_unwrapped_label_error_rad=maximum_label_error,
            exact_event_crops=True,
            original_cached_text_and_pca_preserved=True,
            all_nonturn_rows_preserved=True,
            train_val_test_families_disjoint=True,
            test_turn_events_retained_outside_training=len(held_out),
            backbone_or_single_task_weights_loaded=False,
            source=OmegaConf.to_container(config.source, resolve=True),
        ),
    )
    for name, content in payloads.items():
        (output / f"{name}.json").write_text(
            json.dumps(content, ensure_ascii=False, indent=2) + "\n"
        )
    OmegaConf.save(config, output / "config.yaml", resolve=True)
    print(json.dumps(payloads["audit"], indent=2), flush=True)
