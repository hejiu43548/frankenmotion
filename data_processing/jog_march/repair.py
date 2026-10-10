"""Rebuild jog and march together from category-backed running events."""

from collections import Counter
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import re

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch

from data_processing.event_text import EventTextEncoder
from data_processing.jog_march.rules import CONTEXT
from data_processing.jog_march.rules import RUN_CATEGORIES
from data_processing.jog_march.rules import classify_clip
from data_processing.jog_march.rules import propose_windows
from data_processing.jog_march.rules import semantic_failures
from data_processing.jog_march.rules import trim_to_running_cycles
from scripts.prepare_amass20 import convert_source
from shared_motion.training.catalog import TASK_NAMES
from shared_motion.training.catalog import measure
from shared_motion.training.geometry import Skeleton
from shared_motion.training.model import file_sha256
from shared_motion.training.runner import save_json


def load_candidates(config):
    annotation_path = (
        Path(config.project)
        / "datasets/annotations/frankenstein-dataset/annotations/annotations.json"
    )
    annotations = json.loads(annotation_path.read_text())
    provenance = {str(annotation_path): file_sha256(annotation_path)}
    splits = {}
    for split in ["train", "val", "test"]:
        path = annotation_path.parent / "splits" / f"{split}.txt"
        provenance[str(path)] = file_sha256(path)
        for key in path.read_text().split():
            family = annotations[key]["path"]
            assert splits.setdefault(family, split) == split
    context = defaultdict(list)
    globals_by_family = defaultdict(list)
    candidates = []
    uncertain = []
    for filename in ["train.json", "val.json", "extra_train.json", "extra_val.json"]:
        path = Path(config.babel) / filename
        provenance[str(path)] = file_sha256(path)
        for sequence_id, sequence in json.loads(path.read_text()).items():
            family = str(Path(*Path(sequence["feat_p"]).parts[1:]).with_suffix(""))
            frame_blocks = (
                sequence.get("frame_anns") or []
                if filename.startswith("extra")
                else [sequence.get("frame_ann")]
            )
            seq_blocks = (
                sequence.get("seq_anns") or []
                if filename.startswith("extra")
                else [sequence.get("seq_ann")]
            )
            for block in seq_blocks:
                if block:
                    globals_by_family[family].extend(block["labels"])
            blocks = []
            for block in frame_blocks:
                if not block:
                    continue
                unit = [
                    label
                    for label in block["labels"]
                    if label.get("start_t") == 0 and label.get("end_t") == 1
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
                labels = []
                for label in block["labels"]:
                    if suspicious and label in unit:
                        uncertain.append(
                            dict(
                                family=family,
                                babel_file=filename,
                                babel_sid=sequence_id,
                                label=label,
                                reason="suspicious_shared_0_1s_timing",
                            )
                        )
                    else:
                        labels.append(label)
                context[family].extend(labels)
                blocks.append((dict(block, labels=labels), "frame"))
            if not blocks:
                blocks = [
                    (block, "single_action_sequence")
                    for block in seq_blocks
                    if block
                    and block.get("mul_act") is False
                    and block.get("labels")
                    and all(not semantic_failures(label) for label in block["labels"])
                ]
            for block, level in blocks:
                for original in block["labels"]:
                    if not set(original.get("act_cat") or []) & RUN_CATEGORIES:
                        continue
                    label = dict(original)
                    if level == "single_action_sequence":
                        label.update(start_t=0.0, end_t=sequence["dur"])
                    candidates.append(
                        dict(
                            family=family,
                            split=splits.get(family),
                            babel_file=filename,
                            babel_sid=sequence_id,
                            babel_lid=block.get("babel_lid"),
                            annotation_level=level,
                            source_label=label,
                        )
                    )
    return candidates, context, globals_by_family, provenance, uncertain


@hydra.main(
    version_base="1.3", config_path="../../config", config_name="jog_march_repair"
)
def main(config):
    torch.set_num_threads(2)
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "running_index.json").exists():
        raise ValueError("Completed output exists; choose a new output directory")
    OmegaConf.save(config, output / "config.yaml", resolve=True)
    candidates, context, global_labels, provenance, uncertain = load_candidates(config)
    manifests = {}
    known_sources = {}
    old_tasks = defaultdict(set)
    for split in ["train", "val"]:
        path = Path(config.base) / f"{split}.json"
        manifests[split] = json.loads(path.read_text())
        provenance[str(path)] = file_sha256(path)
        for record in sorted(
            manifests[split], key=lambda entry: entry["task"] not in ["jog", "march"]
        ):
            if record.get("motion_source", "").endswith(".npy"):
                known_sources.setdefault(record["family"], record["motion_source"])
            if record["task"] in ["jog", "march"]:
                old_tasks[record["family"]].add(record["task"])
    repository = Path(__file__).resolve().parents[2]
    for path in list(Path(__file__).parent.glob("*.py")) + [
        repository / "config/jog_march_repair.yaml",
        repository / "shared_motion/training/catalog.py",
        repository / "data_processing/event_text.py",
        Path(config.skeleton),
        Path(config.project) / "outputs_amass/unified_direct_20261008/data/pca.npz",
        Path(config.project) / "pretrained/official/config.json",
    ]:
        provenance[str(path)] = file_sha256(path)
    skeleton = Skeleton(config.skeleton)
    sources = {}
    accepted = []
    rejected = []
    admitted_intervals = defaultdict(list)
    candidates.sort(
        key=lambda record: (
            not record["family"].startswith("MPI_HDM05/"),
            record["annotation_level"] != "frame",
            record["family"],
            record["source_label"]["start_t"],
        )
    )
    for candidate_index, record in enumerate(candidates):
        family = record["family"]
        label = record["source_label"]
        failures = semantic_failures(label)
        if record["split"] not in ["train", "val"]:
            failures.append("test_or_unassigned_source")
        raw = Path(config.amass) / (family + ".npz")
        if not raw.is_file():
            failures.append("missing_raw_source")
        if re.search(r"treadmill|stair|slope|hill|moonwalk", family, re.I) or any(
            re.search(r"treadmill|stair|slope|hill", other.get("proc_label", ""), re.I)
            for other in global_labels[family]
        ):
            failures.append("treadmill_or_uneven_surface_source")
        if failures:
            rejected.append(dict(record, reasons=failures))
            continue
        if family not in sources:
            motion_path = Path(
                known_sources.get(
                    family,
                    str(
                        Path(config.project)
                        / "outputs_amass/unified_direct_20261008/data/motions"
                        / (family + ".npy")
                    ),
                )
            )
            if not motion_path.is_file():
                motion_path = output / "converted_motions" / (family + ".npy")
                result = convert_source(
                    (
                        str(
                            Path(config.project) / "work/unified_direct/prepare_data.py"
                        ),
                        (family, str(raw), str(motion_path), config.skeleton),
                    )
                )
                if result["status"] not in ["ok", "cached"]:
                    rejected.append(
                        dict(record, reasons=["conversion_failed"], conversion=result)
                    )
                    continue
            motion = np.load(motion_path)
            with torch.no_grad():
                joints = skeleton(torch.from_numpy(motion)[None])[0].numpy()
            sources[family] = (motion_path, motion, joints)
            provenance[str(motion_path)] = file_sha256(motion_path)
            provenance[str(raw)] = file_sha256(raw)
        motion_path, motion, joints = sources[family]
        start = max(0, int(np.ceil(label["start_t"] * 20)))
        stop = min(len(motion), int(label["end_t"] * 20))
        times = np.arange(len(motion)) / 20
        valid = np.ones(len(motion), dtype=bool)
        for other in context[family]:
            categories = set(other.get("act_cat") or [])
            if categories - CONTEXT or (
                categories & RUN_CATEGORIES and semantic_failures(other)
            ):
                valid[(times >= other["start_t"]) & (times < other["end_t"])] = False
        windows = propose_windows(joints, start, stop, valid, config.rules)
        if not windows:
            rejected.append(dict(record, reasons=["no_continuous_candidate_bout"]))
        for proposed_task, begin, end in windows:
            trimmed = trim_to_running_cycles(joints[begin:end], config.rules)
            if trimmed is None:
                rejected.append(
                    dict(
                        record,
                        crop_start=begin,
                        crop_stop=end,
                        reasons=["too_short_after_removing_nonrunning_edges"],
                    )
                )
                continue
            original_begin = begin
            begin = original_begin + trimmed[0]
            end = original_begin + trimmed[1]
            clip = motion[begin:end].copy()
            with torch.no_grad():
                crop_joints = skeleton(torch.from_numpy(clip)[None])[0].numpy()
            task, metrics, failures = classify_clip(crop_joints, config.rules)
            if task is not None and task != proposed_task:
                failures.append("inconsistent_window_classification")
            text = label.get("proc_label") or label.get("raw_label") or ""
            if task == "jog" and re.search(
                r"in (?:one )?place|on (?:the )?spot", text, re.I
            ):
                failures.append("in_place_label_but_travelling_motion")
            if task == "march" and re.search(
                r"forward|around|circle|across", text, re.I
            ):
                failures.append("travelling_label_but_stationary_motion")
            if any(
                min(end, prior_end) > max(begin, prior_begin)
                for prior_begin, prior_end in admitted_intervals[family]
            ):
                failures.append("duplicate_or_cross_task_overlap")
            if failures or task is None:
                rejected.append(
                    dict(
                        record,
                        crop_start=begin,
                        crop_stop=end,
                        proposed_task=proposed_task,
                        reasons=failures or ["unclassified"],
                        metrics=metrics,
                    )
                )
                continue
            admitted_intervals[family].append((begin, end))
            key = (
                task
                + "_"
                + hashlib.sha256(f"{family}:{begin}:{end}".encode()).hexdigest()[:20]
            )
            with torch.no_grad():
                value = float(
                    measure(
                        skeleton,
                        torch.from_numpy(clip)[None],
                        torch.tensor([TASK_NAMES.index(task)]),
                        torch.tensor([len(clip)]),
                    )[0]
                )
            accepted.append(
                dict(
                    record,
                    task=task,
                    task_id=TASK_NAMES.index(task),
                    key=key,
                    caption=(
                        "A person runs in place with a steady alternating running gait and stays at the same location."
                        if task == "march"
                        else "A person runs forward with a steady alternating running gait."
                    ),
                    caption_basis="Canonical descriptor after running gait and travel/stationarity checks; original BABEL label retained",
                    original_caption=text,
                    motion_source=str(motion_path),
                    raw_path=str(raw),
                    crop_start_frame_20fps=begin,
                    crop_end_frame_20fps=end,
                    real_frames=len(clip),
                    target_frames=len(clip),
                    pad_frames=0,
                    quantity=value,
                    quantity_definition=(
                        "mean per-foot peak ankle height above fifth-percentile support baseline"
                        if task == "march"
                        else "root XY path length divided by duration"
                    ),
                    semantic_revision=(
                        "stationary_running_v1"
                        if task == "march"
                        else "travelling_running_v1"
                    ),
                    old_task_memberships=sorted(old_tasks[family]),
                    metrics=metrics,
                    source_kind=(
                        "hdm05_babel_running"
                        if family.startswith("MPI_HDM05/")
                        else "babel_running"
                    ),
                    boundary_status="kinematic_bout_within_BABEL_label_not_official_HDM_cuts",
                    manual_verified=False,
                    review_status="awaiting_user_review",
                )
            )
        if (candidate_index + 1) % 100 == 0:
            print(
                json.dumps(dict(processed=candidate_index + 1, accepted=len(accepted))),
                flush=True,
            )
    summary = dict(
        tasks={},
        candidate_events=len(candidates),
        uncertain_time_labels=len(uncertain),
        rejection_reasons=dict(
            Counter(reason for record in rejected for reason in record["reasons"])
        ),
        inspect_only=config.inspect_only,
        training_started=False,
    )
    for task in ["jog", "march"]:
        summary["tasks"][task] = {
            split: dict(
                samples=sum(
                    record["task"] == task and record["split"] == split
                    for record in accepted
                ),
                families=len(
                    {
                        record["family"]
                        for record in accepted
                        if record["task"] == task and record["split"] == split
                    }
                ),
            )
            for split in ["train", "val"]
        }
        summary["tasks"][task]["source_kinds"] = dict(
            Counter(
                record["source_kind"] for record in accepted if record["task"] == task
            )
        )
    save_json(output / "inspection.json", accepted)
    save_json(output / "rejected.json", rejected)
    save_json(output / "uncertain_time_labels.json", uncertain)
    save_json(output / "summary.json", summary)
    save_json(output / "provenance.json", provenance)
    print(json.dumps(summary, indent=2), flush=True)
    if config.inspect_only:
        return
    if any(
        summary["tasks"][task]["train"]["families"] < 10
        or summary["tasks"][task]["val"]["samples"] == 0
        for task in ["jog", "march"]
    ):
        raise ValueError(
            "Insufficient independent training examples or empty validation"
        )
    (output / "cache").mkdir(exist_ok=True)
    encoder = EventTextEncoder(config.project, output)
    for record in accepted:
        begin, end = record["crop_start_frame_20fps"], record["crop_end_frame_20fps"]
        clip = np.load(record["motion_source"], mmap_mode="r")[begin:end].copy()
        text, local, mask = encoder.make_arrays(
            record["caption"],
            np.arange(begin, end),
            begin / 20,
            end / 20,
            ["action", "left_leg", "right_leg"],
        )
        cache = output / "cache" / f'{record["split"]}_{record["key"]}.npz'
        np.savez(
            cache,
            motion=clip,
            tx=text,
            local=local,
            local_mask=mask,
            quantity=np.float32(record["quantity"]),
            task=np.int64(record["task_id"]),
        )
        record.update(
            cache=str(cache), cache_sha256=file_sha256(cache), ready_for_training=True
        )
    combined = {}
    for split, original in manifests.items():
        combined[split] = [
            record for record in original if record["task"] not in ["jog", "march"]
        ] + [record for record in accepted if record["split"] == split]
        assert {record["task"] for record in combined[split]} == set(TASK_NAMES)
        save_json(output / f"{split}.json", combined[split])
    assert not (
        {record["family"] for record in combined["train"]}
        & {record["family"] for record in combined["val"]}
    )
    encoder.save()
    save_json(output / "running_index.json", accepted)


if __name__ == "__main__":
    main()
