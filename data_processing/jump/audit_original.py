#!/usr/bin/env python3
"""Read-only jump inventory, temporal label audit and motion diagnostics."""

from collections import Counter
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import sys

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared_motion.training.geometry import Skeleton
from shared_motion.training.catalog import HUMAN_HEIGHT


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")


def longest_run(mask):
    changes = np.diff(np.pad(np.asarray(mask, dtype=int), (1, 1)))
    lengths = np.flatnonzero(changes == -1) - np.flatnonzero(changes == 1)
    return int(lengths.max()) if len(lengths) else 0


def intervals_overlap(start, stop, event):
    label = event["label"]
    return min(stop, label["end_t"]) > max(start, label["start_t"])


@hydra.main(version_base="1.3", config_path="../../config", config_name="audit_jump")
def main(config):
    torch.set_num_threads(2)
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output / "config.yaml", resolve=True)
    project = Path(config.project)
    annotation_root = project / "datasets/annotations/frankenstein-dataset/annotations"
    annotations = json.loads((annotation_root / "annotations.json").read_text())
    family_splits = {}
    for split in ["train", "val", "test"]:
        for key in (annotation_root / "splits" / f"{split}.txt").read_text().split():
            family = annotations[key]["path"]
            assert family_splits.setdefault(family, split) == split
    all_events = defaultdict(list)
    sequence_labels = defaultdict(list)
    provenance = {"script": digest(__file__), "inputs": {}}
    for filename in ["train.json", "val.json", "extra_train.json", "extra_val.json"]:
        path = Path(config.babel) / filename
        provenance["inputs"][str(path)] = digest(path)
        for sequence_id, sequence in json.loads(path.read_text()).items():
            family = str(Path(*Path(sequence["feat_p"]).parts[1:]).with_suffix(""))
            for level, destination in [("frame", all_events), ("seq", sequence_labels)]:
                blocks = (
                    sequence.get(f"{level}_anns") or []
                    if filename.startswith("extra")
                    else [sequence.get(f"{level}_ann")]
                )
                for block in blocks:
                    if not block:
                        continue
                    for label in block["labels"]:
                        destination[family].append(
                            dict(
                                family=family,
                                split=family_splits.get(family),
                                babel_file=filename,
                                babel_sid=sequence_id,
                                babel_lid=block.get("babel_lid"),
                                level=level,
                                single_action=block.get("mul_act") is False,
                                label=label,
                            )
                        )
    jump_events = {
        family: [
            event
            for event in events
            if {"jump", "hop"} & set(event["label"].get("act_cat") or [])
        ]
        for family, events in all_events.items()
    }
    skeleton = Skeleton(config.skeleton)
    audited = []
    summary = {
        "splits": {},
        "task_definition": "Level-ground two-foot takeoff and two-foot landing; exclude single-leg jumps, platform/stair jumps, jumping jacks and jump rope.",
        "verified_admissible_count": None,
        "diagnostics_are_not_semantic_acceptance": True,
    }
    for split in ["train", "val"]:
        manifest = Path(config.data) / f"{split}.json"
        trained_manifest = Path(config.trained_data) / f"{split}.json"
        provenance["inputs"][str(manifest)] = digest(manifest)
        provenance["inputs"][str(trained_manifest)] = digest(trained_manifest)
        records = [
            record
            for record in json.loads(manifest.read_text())
            if record["task"] == "jump"
        ]
        trained = [
            record
            for record in json.loads(trained_manifest.read_text())
            if record["task"] == "jump"
        ]
        assert (
            records == trained
        ), "Current jump differs from last completed training data"
        flags = Counter()
        quantities = []
        for record in records:
            start = record["crop_start_frame_20fps"] / 20
            stop = start + record["real_frames"] / 20
            events = jump_events.get(record["family"], [])
            overlaps = [
                event for event in events if intervals_overlap(start, stop, event)
            ]
            contained = [
                event
                for event in overlaps
                if start <= event["label"]["start_t"]
                and event["label"]["end_t"] <= stop
            ]
            peak_labels = []
            cache = Path(record["cache"])
            with np.load(cache) as archive:
                motion = torch.from_numpy(
                    archive["motion"][: record["real_frames"]].copy()
                ).float()
                cache_quantity = float(archive["quantity"])
            with torch.no_grad():
                joints = skeleton(motion[None])[0].numpy() * (
                    HUMAN_HEIGHT / skeleton.height
                )
            root_height = joints[:, 0, 2]
            quantity = float(root_height.max() - root_height[0])
            quantities.append(quantity)
            peak_frame = int(root_height.argmax())
            peak_time = start + peak_frame / 20
            for event in all_events.get(record["family"], []):
                label = event["label"]
                if label["start_t"] <= peak_time < label["end_t"]:
                    peak_labels.extend(label.get("act_cat") or [])
            foot_height = np.minimum(joints[:, [7, 8], 2], joints[:, [10, 11], 2])
            foot_lift = foot_height - foot_height.min(axis=0)
            airborne_run = longest_run((foot_lift > 0.06).all(axis=1))
            diagnostic = dict(
                padded=record["pad_frames"] > 0,
                annotation_truncated=record["annotation_end_s"] > stop + 0.051,
                root_rise_below_5cm=quantity < 0.05,
                below_command_range=quantity < 0.25,
                above_command_range=quantity > 0.55,
                no_frame_jump_hop_label_in_family=not events,
                frame_jump_hop_exists_but_crop_misses=bool(events) and not overlaps,
                crop_overlaps_but_contains_no_complete_label=bool(overlaps)
                and not contained,
                no_both_feet_clearance_proxy=airborne_run < 3,
                root_peak_at_crop_boundary=peak_frame in [0, len(motion) - 1],
                peak_has_jump_rope_or_jacks_label=bool(
                    {"jump rope", "jumping jacks"} & set(peak_labels)
                ),
            )
            flags.update(name for name, present in diagnostic.items() if present)
            audited.append(
                dict(
                    record=record,
                    cache_sha256=digest(cache),
                    crop_seconds=[start, stop],
                    quantity_recomputed=quantity,
                    quantity_error=abs(quantity - cache_quantity),
                    peak_frame=peak_frame,
                    peak_categories=sorted(set(peak_labels)),
                    longest_both_feet_clearance_proxy_frames=airborne_run,
                    diagnostic_flags=diagnostic,
                    overlapping_events=overlaps,
                    complete_events=contained,
                    family_jump_hop_events=events,
                )
            )
        summary["splits"][split] = dict(
            records=len(records),
            families=len({record["family"] for record in records}),
            by_dataset=dict(
                Counter(record["family"].split("/")[0] for record in records)
            ),
            flags=dict(flags),
            quantity_quantiles=dict(
                zip(
                    ["min", "p25", "median", "p75", "max"],
                    map(float, np.quantile(quantities, [0, 0.25, 0.5, 0.75, 1])),
                )
            ),
            inside_command_range=sum(0.25 <= value <= 0.55 for value in quantities),
        )
    candidates = []
    seen = set()
    for family, events in jump_events.items():
        for event in events:
            label = event["label"]
            identity = (family, label["start_t"], label["end_t"], label.get("seg_id"))
            if identity in seen:
                continue
            seen.add(identity)
            candidates.append(
                dict(
                    event,
                    raw_available=(Path(config.amass) / (family + ".npz")).is_file(),
                )
            )
    summary["babel_frame_inventory"] = {}
    for split in ["train", "val", "test", None]:
        selected = [
            event
            for event in candidates
            if event["split"] == split and event["raw_available"]
        ]
        hdm = [event for event in selected if event["family"].startswith("MPI_HDM05/")]
        clean_category = [
            event
            for event in selected
            if not {"jump rope", "jumping jacks"} & set(event["label"]["act_cat"])
        ]
        summary["babel_frame_inventory"][str(split)] = dict(
            events=len(selected),
            families=len({event["family"] for event in selected}),
            hdm_events=len(hdm),
            hdm_families=len({event["family"] for event in hdm}),
            excluding_explicit_rope_jacks_events=len(clean_category),
            excluding_explicit_rope_jacks_families=len(
                {event["family"] for event in clean_category}
            ),
        )
    hdm_sources = []
    for raw in sorted((Path(config.amass) / "MPI_HDM05").glob("*/*.npz")):
        family = str(raw.relative_to(config.amass).with_suffix(""))
        labels = [
            event
            for event in sequence_labels.get(family, [])
            if {"jump", "hop"} & set(event["label"].get("act_cat") or [])
        ]
        events = jump_events.get(family, [])
        if labels or events:
            hdm_sources.append(
                dict(
                    family=family,
                    split=family_splits.get(family),
                    raw=str(raw),
                    frame_events=events,
                    sequence_labels=labels,
                )
            )
    summary["hdm_with_jump_hop_frame_or_sequence_labels"] = dict(
        Counter(str(record["split"]) for record in hdm_sources)
    )
    summary["max_cache_quantity_error"] = max(
        record["quantity_error"] for record in audited
    )
    summary["unchanged_since_last_training"] = True
    summary["limitations"] = [
        "BABEL jump/hop category alone does not certify two-foot vertical jumps; variants and concurrent actions require review.",
        "Frame counts deduplicate family/start/end/seg_id; overlapping annotator events are not independent clips.",
        "Missing BABEL annotation is missing evidence, not proof of a wrong motion.",
        "Both-feet clearance uses a 6cm relative foot minimum and 3 consecutive frames; it is a diagnostic proxy, not contact ground truth.",
        "HDM05 official cuts endpoint returned HTTP403; no native cut boundaries or hopBothLegs sample count verified.",
    ]
    save(output / "summary.json", summary)
    save(output / "current_jump_audit.json", audited)
    save(output / "babel_frame_candidates.json", candidates)
    save(output / "hdm_sources.json", hdm_sources)
    save(output / "provenance.json", provenance)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
