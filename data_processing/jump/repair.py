"""Select category-backed jump events, crop actual flights and keep provenance."""

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
from data_processing.jump.rules import ALLOWED_CONTEXT_CATEGORIES
from data_processing.jump.rules import DIRECTIONAL_JUMP
from data_processing.jump.rules import check_clip
from data_processing.jump.rules import propose_windows
from data_processing.jump.rules import semantic_failures
from scripts.prepare_amass20 import convert_source
from shared_motion.training.catalog import HUMAN_HEIGHT
from shared_motion.training.catalog import TASK_NAMES
from shared_motion.training.catalog import measure
from shared_motion.training.geometry import Skeleton
from shared_motion.training.model import file_sha256
from shared_motion.training.runner import save_json


def load_candidates(config):
    annotation_root = (
        Path(config.project) / "datasets/annotations/frankenstein-dataset/annotations"
    )
    annotation_path = annotation_root / "annotations.json"
    annotations = json.loads(annotation_path.read_text())
    provenance = {str(annotation_path): file_sha256(annotation_path)}
    family_splits = {}
    for split in ["train", "val", "test"]:
        path = annotation_root / "splits" / f"{split}.txt"
        provenance[str(path)] = file_sha256(path)
        for key in path.read_text().split():
            family = annotations[key]["path"]
            assert family_splits.setdefault(family, split) == split
    events = []
    context = defaultdict(list)
    global_labels = defaultdict(list)
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
            sequence_blocks = (
                sequence.get("seq_anns") or []
                if filename.startswith("extra")
                else [sequence.get("seq_ann")]
            )
            for block in sequence_blocks:
                if block:
                    global_labels[family].extend(block["labels"])
            blocks = [(block, "frame") for block in frame_blocks if block]
            for block, _ in blocks:
                context[family].extend(block["labels"])
            if not blocks:
                blocks = [
                    (block, "single_action_sequence")
                    for block in sequence_blocks
                    if block
                    and block.get("mul_act") is False
                    and block.get("labels")
                    and all(not semantic_failures(label) for label in block["labels"])
                ]
            for block, level in blocks:
                for label in block["labels"]:
                    if not {"jump", "hop"} & set(label.get("act_cat") or []):
                        continue
                    label = dict(label)
                    if level == "single_action_sequence":
                        label.update(start_t=0.0, end_t=sequence["dur"])
                    events.append(
                        dict(
                            family=family,
                            split=family_splits.get(family),
                            babel_file=filename,
                            babel_sid=sequence_id,
                            babel_lid=block.get("babel_lid"),
                            source_label=label,
                            annotation_level=level,
                        )
                    )
    return events, context, global_labels, provenance


@hydra.main(version_base="1.3", config_path="../../config", config_name="jump_repair")
def main(config):
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "jump_index.json").exists():
        raise ValueError("Finalized output already exists; use a new version")
    OmegaConf.save(config, output / "config.yaml", resolve=True)
    torch.set_num_threads(2)
    skeleton = Skeleton(config.skeleton)
    candidates, context, global_labels, provenance = load_candidates(config)
    provenance[str(config.skeleton)] = file_sha256(config.skeleton)
    for path in [
        Path(config.project) / "outputs_amass/unified_direct_20261008/data/pca.npz",
        Path(config.project) / "pretrained/official/config.json",
        Path(config.project) / "work/unified_direct/prepare_data.py",
        Path(__file__).parents[1] / "event_text.py",
        Path(__file__).resolve().parents[2] / "config/jump_repair.yaml",
    ]:
        provenance[str(path)] = file_sha256(path)
    for source in sorted(Path(__file__).parent.glob("*.py")):
        provenance[str(source)] = file_sha256(source)
    candidates.sort(
        key=lambda record: (
            not record["family"].startswith("MPI_HDM05/"),
            record["annotation_level"] != "frame",
            record["family"],
            record["source_label"]["start_t"],
        )
    )
    accepted = []
    rejected = []
    seen_windows = defaultdict(list)
    source_cache = {}
    for candidate_index, record in enumerate(candidates):
        family = record["family"]
        label = record["source_label"]
        failures = semantic_failures(label)
        if record["split"] not in ["train", "val"]:
            failures.append("heldout_or_unassigned_family")
        raw = Path(config.amass) / (family + ".npz")
        if not raw.is_file():
            failures.append("missing_raw_source")
        hard_variants = re.compile(
            r"rope|jack|bench|stair|platform|obstacle|barrier|hurdle|vault|table|box",
            re.I,
        )
        if hard_variants.search(family) or any(
            hard_variants.search(annotation.get("proc_label", ""))
            for annotation in global_labels[family]
        ):
            failures.append("source_has_object_height_or_rope_jacks_evidence")
        if any(
            {"jump", "hop"} & set(annotation.get("act_cat") or [])
            and DIRECTIONAL_JUMP.search(
                " ".join(
                    annotation.get(name) or "" for name in ["proc_label", "raw_label"]
                )
            )
            for annotation in global_labels[family]
        ):
            failures.append("source_has_directional_jump_sequence_label")
        if failures:
            rejected.append(dict(record, reasons=failures))
            continue
        if family not in source_cache:
            motion_path = (
                Path(config.project)
                / "outputs_amass/unified_direct_20261008/data/motions"
                / (family + ".npy")
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
            full_motion = np.load(motion_path)
            with torch.no_grad():
                joints = skeleton(torch.from_numpy(full_motion)[None])[0].numpy() * (
                    HUMAN_HEIGHT / skeleton.height
                )
            source_cache[family] = (motion_path, full_motion, joints)
            provenance[str(motion_path)] = file_sha256(motion_path)
            provenance[str(raw)] = file_sha256(raw)
        motion_path, full_motion, joints = source_cache[family]
        start = max(0, int(label["start_t"] * 20))
        stop = min(len(full_motion), int(np.ceil(label["end_t"] * 20)))
        windows = propose_windows(joints, start, stop, config.rules)
        if not windows:
            rejected.append(
                dict(record, reasons=["no_complete_bilateral_flight_window"])
            )
        for begin, end, flight_start, flight_stop in windows:
            # Actual crop FK reanchors yaw/XY; all tests use these exact saved frames.
            clip = full_motion[begin:end].copy()
            with torch.no_grad():
                crop_joints = skeleton(torch.from_numpy(clip)[None])[0].numpy() * (
                    HUMAN_HEIGHT / skeleton.height
                )
            metrics, failures = check_clip(crop_joints, config.rules)
            conflicting = []
            for other in context[family]:
                overlap = min(end / 20, other["end_t"]) - max(
                    begin / 20, other["start_t"]
                )
                if overlap > 0.10 and (
                    set(other.get("act_cat") or []) - ALLOWED_CONTEXT_CATEGORIES
                    or (
                        {"jump", "hop"} & set(other.get("act_cat") or [])
                        and semantic_failures(other)
                    )
                ):
                    conflicting.append(other)
            if conflicting:
                failures.append("conflicting_frame_category_inside_crop")
            if any(
                min(flight_stop, prior_stop) > max(flight_start, prior_start)
                for prior_start, prior_stop in seen_windows[family]
            ):
                failures.append("duplicate_physical_flight")
            if failures:
                rejected.append(
                    dict(
                        record,
                        crop_start=begin,
                        crop_stop=end,
                        reasons=failures,
                        metrics=metrics,
                        conflicting_labels=conflicting,
                    )
                )
                continue
            seen_windows[family].append((flight_start, flight_stop))
            identity = f"{family}:{begin}:{end}"
            key = "jump_" + hashlib.sha256(identity.encode()).hexdigest()[:20]
            accepted.append(
                dict(
                    record,
                    key=key,
                    task="jump",
                    task_id=TASK_NAMES.index("jump"),
                    motion_source=str(motion_path),
                    raw_path=str(raw),
                    crop_start_frame_20fps=begin,
                    crop_end_frame_20fps=end,
                    flight_start_frame_20fps=flight_start,
                    flight_end_frame_20fps=flight_stop,
                    real_frames=len(clip),
                    target_frames=len(clip),
                    pad_frames=0,
                    metrics=metrics,
                    source_kind=(
                        "hdm05_babel_kinematic_jump"
                        if family.startswith("MPI_HDM05/")
                        else "babel_kinematic_jump"
                    ),
                    boundary_status="kinematic_flight_and_support_crop_not_official_hdm_cuts",
                    original_caption=label.get("proc_label") or label.get("raw_label"),
                    caption="A person jumps straight up in place with both feet and lands at the same spot on level ground.",
                    caption_basis="Canonical descriptor after category and kinematic admission; original event label retained",
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
        sources=dict(Counter(record["source_kind"] for record in accepted)),
        annotation_levels=dict(
            Counter(record["annotation_level"] for record in accepted)
        ),
        rejection_reasons=dict(
            Counter(reason for record in rejected for reason in record["reasons"])
        ),
        candidate_events=len(candidates),
        inspect_only=config.inspect_only,
        trained=False,
        official_hdm_cuts_verified=False,
    )
    save_json(output / "inspection.json", accepted)
    save_json(output / "rejected.json", rejected)
    save_json(output / "summary.json", summary)
    save_json(output / "provenance.json", provenance)
    print(json.dumps(summary, indent=2), flush=True)
    if config.inspect_only:
        return
    if summary["families"]["train"] < 10 or not summary["counts"]["val"]:
        raise ValueError(
            "Insufficient TRAIN families or empty validation; do not fabricate samples"
        )
    encoder = EventTextEncoder(config.project, output)
    (output / "cache").mkdir(exist_ok=True)
    for record in accepted:
        begin = record["crop_start_frame_20fps"]
        end = record["crop_end_frame_20fps"]
        clip = np.load(record["motion_source"], mmap_mode="r")[begin:end].copy()
        with torch.no_grad():
            quantity = float(
                measure(
                    skeleton,
                    torch.from_numpy(clip)[None],
                    torch.tensor([record["task_id"]]),
                    torch.tensor([len(clip)]),
                )[0]
            )
        global_text, local_text, local_mask = encoder.make_arrays(
            record["caption"],
            np.arange(begin, end),
            begin / 20,
            end / 20,
            ["action", "left_leg", "right_leg"],
        )
        cache_path = output / "cache" / f'{record["split"]}_{record["key"]}.npz'
        np.savez(
            cache_path,
            motion=clip,
            tx=global_text,
            local=local_text,
            local_mask=local_mask,
            quantity=np.float32(quantity),
            task=np.int64(record["task_id"]),
        )
        record.update(
            quantity=quantity,
            cache=str(cache_path),
            cache_sha256=file_sha256(cache_path),
            ready_for_training=True,
        )
    manifests = {}
    for split in ["train", "val"]:
        source = Path(config.base) / f"{split}.json"
        original = json.loads(source.read_text())
        provenance[str(source)] = file_sha256(source)
        manifests[split] = [
            record for record in original if record["task"] != "jump"
        ] + [record for record in accepted if record["split"] == split]
        assert {record["task"] for record in manifests[split]} == set(TASK_NAMES)
        save_json(output / f"{split}.json", manifests[split])
    assert not (
        {record["family"] for record in manifests["train"]}
        & {record["family"] for record in manifests["val"]}
    )
    save_json(output / "jump_index.json", accepted)
    save_json(output / "provenance.json", provenance)
    encoder.save()


if __name__ == "__main__":
    main()
