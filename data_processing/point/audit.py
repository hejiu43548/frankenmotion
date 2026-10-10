"""Audit point captions against category timing and actual wrist/root motion."""

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


def load_labels(config):
    annotations = json.loads(
        (
            Path(config.project)
            / "datasets/annotations/frankenstein-dataset/annotations/annotations.json"
        ).read_text()
    )
    split_root = (
        Path(config.project)
        / "datasets/annotations/frankenstein-dataset/annotations/splits"
    )
    splits = {}
    for split in ["train", "val", "test"]:
        for key in (split_root / f"{split}.txt").read_text().split():
            splits[annotations[key]["path"]] = split
    frames = defaultdict(list)
    sequences = defaultdict(list)
    uncertain = []
    candidates = []
    categories = Counter()
    for filename in ["train.json", "val.json", "extra_train.json", "extra_val.json"]:
        path = Path(config.babel) / filename
        for sequence_id, sequence in json.loads(path.read_text()).items():
            family = str(Path(*Path(sequence["feat_p"]).parts[1:]).with_suffix(""))
            for level, target in [("frame", frames), ("seq", sequences)]:
                blocks = (
                    sequence.get(level + "_anns") or []
                    if filename.startswith("extra")
                    else [sequence.get(level + "_ann")]
                )
                for block in blocks:
                    if not block:
                        continue
                    unit = [
                        label
                        for label in block["labels"]
                        if level == "frame"
                        and label.get("start_t") == 0
                        and label.get("end_t") == 1
                    ]
                    suspicious = (
                        sequence["dur"] > 5
                        and len(unit) >= 4
                        and len(
                            {
                                category
                                for label in unit
                                for category in label.get("act_cat") or []
                            }
                        )
                        >= 4
                    )
                    for label in block["labels"]:
                        record = dict(
                            family=family,
                            split=splits.get(family),
                            babel_file=filename,
                            babel_sid=sequence_id,
                            annotation_level=level,
                            single_action=block.get("mul_act") is False,
                            label=label,
                            duration=sequence["dur"],
                        )
                        if suspicious and label in unit:
                            uncertain.append(record)
                            continue
                        target[family].append(record)
                        categories.update(label.get("act_cat") or [])
                        if set(label.get("act_cat") or []) & {"point"}:
                            candidates.append(record)
    return frames, sequences, candidates, uncertain, categories


@hydra.main(version_base="1.3", config_path="../../config", config_name="audit_point")
def main(config):
    torch.set_num_threads(2)
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output / "config.yaml", resolve=True)
    frames, sequences, candidates, uncertain, categories = load_labels(config)
    skeleton = Skeleton(config.skeleton)
    audited = []
    summary = dict(splits={}, training_started=False, manifests_modified=False)
    provenance = {str(Path(__file__).resolve()): file_sha256(__file__)}
    location_pattern = re.compile(
        r"(?:start(?:ing)?|end(?:ing)?|original|same|different|another|other|given|certain|initial) point|point [ab]\b|one point to|point to point",
        re.I,
    )
    for split in ["train", "val"]:
        path = Path(config.data) / f"{split}.json"
        provenance[str(path)] = file_sha256(path)
        rows = [
            record
            for record in json.loads(path.read_text())
            if record["task"] == "point"
        ]
        selected = []
        for record in rows:
            with np.load(record["cache"]) as archive:
                motion = archive["motion"][: record["real_frames"]].copy()
            assert file_sha256(record["cache"]) == record["cache_sha256"]
            begin, end = (
                record["crop_start_frame_20fps"],
                record["crop_end_frame_20fps"],
            )
            source = np.load(record["motion_source"], mmap_mode="r")
            assert np.array_equal(motion, source[begin:end])
            with torch.no_grad():
                joints = skeleton(torch.from_numpy(motion)[None])[0].numpy()
            times = np.arange(begin, end) / 20
            overlaps = [
                event
                for event in frames[record["family"]]
                if min(end / 20, event["label"]["end_t"])
                > max(begin / 20, event["label"]["start_t"])
            ]
            point_events = [
                event
                for event in frames[record["family"]]
                if "point" in (event["label"].get("act_cat") or [])
            ]
            mask = np.zeros(len(times), dtype=bool)
            for event in point_events:
                mask |= (times >= event["label"]["start_t"]) & (
                    times < event["label"]["end_t"]
                )
            wrist = joints[:, 21] - joints[:, 0]
            root = joints[:, 0]
            metrics = dict(
                point_frame_coverage=float(mask.mean()),
                frame_point_events_in_source=len(point_events),
                no_overlap_despite_point_event=bool(point_events and not mask.any()),
                location_word_match=bool(location_pattern.search(record["caption"])),
                root_excursion_m=float(
                    np.linalg.norm(root[:, :2] - root[:1, :2], axis=1).max()
                ),
                right_wrist_range_xyz_m=np.ptp(wrist, axis=0).tolist(),
                right_wrist_max_forward_m=float(wrist[:, 0].max()),
                caption=record["caption"],
            )
            selected.append(
                dict(
                    record,
                    audit=metrics,
                    overlapping_labels=overlaps,
                    sequence_labels=sequences[record["family"]],
                )
            )
        summary["splits"][split] = dict(
            samples=len(rows),
            families=len({record["family"] for record in rows}),
            zero_frame_point_coverage=sum(
                record["audit"]["point_frame_coverage"] == 0 for record in selected
            ),
            missed_point_event=sum(
                record["audit"]["no_overlap_despite_point_event"] for record in selected
            ),
            location_word_match=sum(
                record["audit"]["location_word_match"] for record in selected
            ),
            moving_over_30cm=sum(
                record["audit"]["root_excursion_m"] > 0.3 for record in selected
            ),
            overlapping_categories=dict(
                Counter(
                    category
                    for record in selected
                    for event in record["overlapping_labels"]
                    for category in event["label"].get("act_cat") or []
                )
            ),
        )
        audited.extend(selected)
    summary["candidate_inventory"] = {
        split: {
            level: dict(
                events=sum(
                    record["split"] == split and record["annotation_level"] == level
                    for record in candidates
                ),
                families=len(
                    {
                        record["family"]
                        for record in candidates
                        if record["split"] == split
                        and record["annotation_level"] == level
                    }
                ),
            )
            for level in ["frame", "seq"]
        }
        for split in ["train", "val", "test", None]
    }
    summary["point_like_category_names"] = {
        name: count
        for name, count in categories.items()
        if re.search("point|reach|grab", name)
    }
    summary["uncertain_timing_labels"] = len(uncertain)
    save_json(output / "records.json", audited)
    save_json(output / "candidates.json", candidates)
    save_json(output / "uncertain_timing.json", uncertain)
    save_json(output / "summary.json", summary)
    save_json(output / "provenance.json", provenance)
    print(json.dumps(summary, indent=2), flush=True)
    examples = [record for record in audited if record["audit"]["location_word_match"]]
    print(
        json.dumps(
            [
                dict(family=record["family"], caption=record["caption"])
                for record in examples[:10]
            ],
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
