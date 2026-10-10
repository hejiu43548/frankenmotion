"""Audit actual jog clips against source timing, categories and movement."""

from collections import Counter
from collections import defaultdict
import json
from pathlib import Path
import re

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch

from shared_motion.training.geometry import Skeleton
from shared_motion.training.model import file_sha256
from shared_motion.training.runner import save_json


def label_mask(times, events, categories):
    selected = np.zeros(len(times), dtype=bool)
    for event in events:
        label = event["label"]
        if set(label.get("act_cat") or []) & categories:
            selected |= (times >= label["start_t"]) & (times < label["end_t"])
    return selected


def movement_metrics(joints):
    root = joints[:, 0]
    velocity = np.diff(root[:, :2], axis=0) * 20
    speeds = np.linalg.norm(velocity, axis=1)
    lateral = joints[:, 1, :2] - joints[:, 2, :2]
    lateral /= np.linalg.norm(lateral, axis=1, keepdims=True).clip(1e-8)
    forward = np.stack([lateral[:, 1], -lateral[:, 0]], axis=1)
    projected = (velocity * forward[:-1]).sum(axis=1)
    foot_height = np.minimum(joints[:, [7, 8], 2], joints[:, [10, 11], 2])
    clearance = foot_height - np.quantile(foot_height, 0.1, axis=0)
    flight = (clearance > 0.03).all(axis=1)
    changes = np.diff(np.pad(flight.astype(int), (1, 1)))
    durations = np.flatnonzero(changes == -1) - np.flatnonzero(changes == 1)
    torso = (joints[:, 16] + joints[:, 17]) / 2 - root
    foot_forward = ((joints[:, [7, 8], :2] - root[:, None, :2]) * forward[:, None]).sum(
        axis=-1
    )
    return dict(
        path_speed_m_s=float(speeds.mean()),
        net_displacement_m=float(np.linalg.norm(root[-1, :2] - root[0, :2])),
        root_excursion_m=float(
            np.linalg.norm(root[:, :2] - root[:1, :2], axis=1).max()
        ),
        near_stationary_fraction=float((speeds < 0.1).mean()),
        speed_p10_m_s=float(np.quantile(speeds, 0.1)),
        forward_speed_m_s=float(projected.mean()),
        backward_motion_fraction=float((projected < -0.2).mean()),
        airborne_proxy_fraction=float(flight.mean()),
        sustained_airborne_proxy_runs=int((durations >= 2).sum()),
        ankle_swing_range_m=np.ptp(foot_forward, axis=0).tolist(),
        maximum_torso_tilt_rad=float(
            np.arctan2(np.linalg.norm(torso[:, :2], axis=1), torso[:, 2]).max()
        ),
    )


@hydra.main(version_base="1.3", config_path="../../config", config_name="audit_jog")
def main(config):
    torch.set_num_threads(2)
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output / "config.yaml", resolve=True)
    annotation_path = (
        Path(config.project)
        / "datasets/annotations/frankenstein-dataset/annotations/annotations.json"
    )
    annotations = json.loads(annotation_path.read_text())
    provenance = {
        str(annotation_path): file_sha256(annotation_path),
        str(config.skeleton): file_sha256(config.skeleton),
        str(Path(__file__).resolve()): file_sha256(__file__),
    }
    family_split = {}
    for split in ["train", "val", "test"]:
        path = annotation_path.parent / "splits" / f"{split}.txt"
        for key in path.read_text().split():
            family = annotations[key]["path"]
            assert family_split.setdefault(family, split) == split
        provenance[str(path)] = file_sha256(path)
    frames = defaultdict(list)
    sequences = defaultdict(list)
    uncertain_timing = []
    for filename in ["train.json", "val.json", "extra_train.json", "extra_val.json"]:
        path = Path(config.babel) / filename
        provenance[str(path)] = file_sha256(path)
        for sequence_id, sequence in json.loads(path.read_text()).items():
            family = str(Path(*Path(sequence["feat_p"]).parts[1:]).with_suffix(""))
            for level, target in [("frame", frames), ("seq", sequences)]:
                blocks = (
                    sequence.get(level + "_anns") or []
                    if filename.startswith("extra")
                    else [sequence.get(level + "_ann")]
                )
                for block in blocks:
                    if block:
                        unit_intervals = [
                            label
                            for label in block["labels"]
                            if level == "frame"
                            and label.get("start_t") == 0
                            and label.get("end_t") == 1
                        ]
                        suspect_timing = (
                            sequence["dur"] > 5
                            and len(unit_intervals) >= 4
                            and len(
                                {
                                    category
                                    for label in unit_intervals
                                    for category in label.get("act_cat") or []
                                }
                            )
                            >= 4
                        )
                        for label in block["labels"]:
                            if suspect_timing and label in unit_intervals:
                                uncertain_timing.append(
                                    dict(
                                        family=family,
                                        babel_file=filename,
                                        babel_sid=sequence_id,
                                        babel_lid=block.get("babel_lid"),
                                        duration_s=sequence["dur"],
                                        label=label,
                                        reason="Many distinct actions all carry exact0..1s intervals in a long sequence; timing not trusted for this audit",
                                    )
                                )
                                continue
                            target[family].append(
                                dict(
                                    family=family,
                                    babel_file=filename,
                                    babel_sid=sequence_id,
                                    babel_lid=block.get("babel_lid"),
                                    single_action=block.get("mul_act") is False,
                                    label=label,
                                )
                            )
    skeleton = Skeleton(config.skeleton)
    audited = []
    summary = dict(
        splits={},
        data=str(config.data),
        training_started=False,
        manifests_modified=False,
    )
    pattern = re.compile(r"\bjog|\brun(?:ning)?\b", re.I)
    for split in ["train", "val"]:
        path = Path(config.data) / f"{split}.json"
        trained_path = Path(config.trained_data) / f"{split}.json"
        records = [
            record for record in json.loads(path.read_text()) if record["task"] == "jog"
        ]
        trained = [
            record
            for record in json.loads(trained_path.read_text())
            if record["task"] == "jog"
        ]
        assert records == trained
        provenance[str(path)] = file_sha256(path)
        provenance[str(trained_path)] = file_sha256(trained_path)
        split_rows = []
        for record in records:
            with np.load(record["cache"]) as archive:
                motion = archive["motion"][: record["real_frames"]].copy()
                quantity = float(archive["quantity"])
            digest = file_sha256(record["cache"])
            assert digest == record["cache_sha256"]
            source = np.load(record["motion_source"], mmap_mode="r")
            start, stop = (
                record["crop_start_frame_20fps"],
                record["crop_end_frame_20fps"],
            )
            assert np.array_equal(motion, source[start:stop])
            with torch.no_grad():
                joints = skeleton(torch.from_numpy(motion)[None])[0].numpy()
            metrics = movement_metrics(joints)
            times = np.arange(start, stop) / 20
            events = frames[record["family"]]
            run_mask = label_mask(times, events, {"run", "jog"})
            walk_mask = label_mask(times, events, {"walk"})
            stand_mask = label_mask(times, events, {"stand"})
            incompatible_mask = label_mask(
                times,
                events,
                {
                    "jump",
                    "hop",
                    "dance",
                    "kick",
                    "punch",
                    "sit",
                    "crawl",
                    "jumping jacks",
                    "jump rope",
                    "squat",
                },
            )
            all_mask = label_mask(
                times,
                events,
                {
                    category
                    for event in events
                    for category in event["label"].get("act_cat") or []
                },
            )
            overlapping = [
                event
                for event in events
                if min(stop / 20, event["label"]["end_t"])
                > max(start / 20, event["label"]["start_t"])
            ]
            family_run = [
                event
                for event in events
                if {"run", "jog"} & set(event["label"].get("act_cat") or [])
            ]
            annotation = annotations[record["annotation_key"]]
            matching_parts = [
                segment
                for segment in annotation["annotations"]
                if segment["bodypart"] != "sequence_caption"
                and pattern.search(segment["text"])
                and min(stop / 20, segment["end"]) > max(start / 20, segment["start"])
            ]
            coverage = dict(
                run_jog=float(run_mask.mean()),
                walk=float(walk_mask.mean()),
                stand=float(stand_mask.mean()),
                incompatible=float(incompatible_mask.mean()),
                any_frame_category=float(all_mask.mean()),
            )
            flags = dict(
                below_command_range=quantity < 0.8,
                above_command_range=quantity > 2,
                little_root_travel=metrics["path_speed_m_s"] < 0.2,
                substantial_pause=metrics["near_stationary_fraction"] > 0.25,
                backward_motion=metrics["backward_motion_fraction"] > 0.25,
                no_frame_run_jog_evidence_in_family=not family_run,
                frame_run_jog_exists_but_crop_misses=bool(family_run)
                and not run_mask.any(),
                walk_without_run_for_majority=float((walk_mask & ~run_mask).mean())
                >= 0.5,
                stand_without_run_for_majority=float((stand_mask & ~run_mask).mean())
                >= 0.5,
                incompatible_without_run_at_least_quarter=float(
                    (incompatible_mask & ~run_mask).mean()
                )
                >= 0.25,
                run_jog_coverage_below_half=bool(family_run)
                and coverage["run_jog"] < 0.5,
                no_sustained_flight_proxy=metrics["sustained_airborne_proxy_runs"] == 0,
                source_full_caption_fallback="global-caption match"
                in record["semantic_event_verification"],
                no_action_matching_segment=bool(matching_parts)
                and not any(
                    segment["bodypart"] == "action" for segment in matching_parts
                ),
            )
            row = dict(
                record=record,
                metrics=metrics,
                coverage=coverage,
                flags=flags,
                quantity_error=abs(metrics["path_speed_m_s"] - quantity),
                matching_frankenstein_segments=matching_parts,
                overlapping_babel_events=overlapping,
                family_run_jog_events=family_run,
                sequence_labels=sequences[record["family"]],
            )
            audited.append(row)
            split_rows.append(row)
        summary["splits"][split] = dict(
            records=len(records),
            families=len({record["family"] for record in records}),
            datasets=dict(
                Counter(record["family"].split("/")[0] for record in records)
            ),
            flags=dict(
                Counter(
                    name
                    for row in split_rows
                    for name, present in row["flags"].items()
                    if present
                )
            ),
            inside_command_range=sum(
                0.8 <= row["record"]["quantity"] <= 2 for row in split_rows
            ),
            run_jog_frame_coverage_at_least_80pct=sum(
                row["coverage"]["run_jog"] >= 0.8 for row in split_rows
            ),
            speed_quantiles=dict(
                zip(
                    ["min", "p25", "median", "p75", "max"],
                    map(
                        float,
                        np.quantile(
                            [row["metrics"]["path_speed_m_s"] for row in split_rows],
                            [0, 0.25, 0.5, 0.75, 1],
                        ),
                    ),
                )
            ),
        )
    candidates = []
    seen = set()
    for family, events in frames.items():
        for event in events:
            label = event["label"]
            if not {"run", "jog"} & set(label.get("act_cat") or []):
                continue
            identity = (family, label["start_t"], label["end_t"], label.get("seg_id"))
            if identity in seen:
                continue
            seen.add(identity)
            candidates.append(
                dict(
                    event,
                    split=family_split.get(family),
                    raw_available=(Path(config.amass) / (family + ".npz")).is_file(),
                )
            )
    summary["babel_frame_inventory"] = {}
    for split in ["train", "val", "test", None]:
        selected = [
            record
            for record in candidates
            if record["split"] == split and record["raw_available"]
        ]
        hdm = [
            record for record in selected if record["family"].startswith("MPI_HDM05/")
        ]
        summary["babel_frame_inventory"][str(split)] = dict(
            events=len(selected),
            families=len({record["family"] for record in selected}),
            hdm_events=len(hdm),
            hdm_families=len({record["family"] for record in hdm}),
        )
    summary["max_quantity_error"] = max(row["quantity_error"] for row in audited)
    assert summary["max_quantity_error"] < 1e-5
    summary["all_source_crops_exact"] = True
    summary["unchanged_since_completed_training"] = True
    summary["uncertain_timing_labels_excluded_from_temporal_counts"] = len(
        uncertain_timing
    )
    summary["uncertain_timing_families"] = len(
        {event["family"] for event in uncertain_timing}
    )
    summary["limitations"] = [
        "Frame categories may be incomplete or conflicting; missing evidence alone is not a wrong-motion verdict.",
        "Airborne status is only a foot-height proxy, not measured contact truth or a sufficient running classifier.",
        "Only existing raw files and labels are inventoried; no official HDM05 cut mapping is assumed.",
    ]
    save_json(output / "summary.json", summary)
    save_json(output / "current_jog_audit.json", audited)
    save_json(output / "babel_frame_candidates.json", candidates)
    save_json(output / "provenance.json", provenance)
    save_json(output / "uncertain_babel_timing.json", uncertain_timing)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
