"""Merge point/reach into verified right-wrist approach-and-hold XYZ events."""

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
from data_processing.point.audit import load_labels
from scripts.prepare_amass20 import convert_source
from shared_motion.training.geometry import Skeleton
from shared_motion.training.model import file_sha256
from shared_motion.training.reach import REACH_REVISION
from shared_motion.training.reach import target_xyz
from shared_motion.training.reach import wrist_coordinates
from shared_motion.training.runner import save_json

CATEGORIES = {
    "point",
    "grasp object",
    "touch object",
    "interact with/use object",
    "hand movements",
    "arm movements",
    "stretch",
    "raising body part",
    "lowering body part",
}
CONTEXT = CATEGORIES | {"stand", "transition", "look", "wait", "stop"}
BAD_TEXT = re.compile(
    r"\bleft\b|both|two hands|floor|ground|bend|kneel|sit|walk|step|punch|wave|throw|dance|gun|rifle|scratch|head|face|body part|clap|carry|lift|basket|ball|starting point|original point|point [ab]\b",
    re.I,
)
REACH_TEXT = re.compile(r"\breach|\bgrab|\bgrasp", re.I)


def compatible_event(event):
    label = event["label"]
    categories = set(label.get("act_cat") or [])
    text = label.get("proc_label") or label.get("raw_label") or ""
    if BAD_TEXT.search(text) or categories - CONTEXT:
        return False
    return bool(
        "point" in categories or (categories & CATEGORIES and REACH_TEXT.search(text))
    )


def crop_metrics(joints, rules):
    right = joints[:, 21]
    left = joints[:, 20]
    hold = right[-rules.hold_frames :]
    right_displacement = np.linalg.norm(hold.mean(0) - right[0])
    left_displacement = np.linalg.norm(left - left[0], axis=1).max()
    root = joints[:, 0]
    torso = (joints[:, 16] + joints[:, 17]) / 2 - root
    upper = joints[-rules.hold_frames :, 17] - joints[-rules.hold_frames :, 19]
    lower = joints[-rules.hold_frames :, 21] - joints[-rules.hold_frames :, 19]
    elbow = np.rad2deg(
        np.arccos(
            np.clip(
                (upper * lower).sum(-1)
                / (
                    np.linalg.norm(upper, axis=-1) * np.linalg.norm(lower, axis=-1)
                ).clip(1e-8),
                -1,
                1,
            )
        )
    ).mean()
    lateral = joints[:, 1, :2] - joints[:, 2, :2]
    heading = np.unwrap(np.arctan2(lateral[:, 1], lateral[:, 0]))
    metrics = dict(
        hold_span_m=float(np.linalg.norm(hold - hold.mean(0), axis=-1).max()),
        hold_speed_m_s=float(
            np.linalg.norm(np.diff(hold, axis=0), axis=-1).mean() * 20
        ),
        right_wrist_displacement_m=float(right_displacement),
        left_wrist_displacement_m=float(left_displacement),
        arm_extension_m=float(
            np.linalg.norm(
                (
                    joints[-rules.hold_frames :, 21] - joints[-rules.hold_frames :, 17]
                ).mean(0)
            )
        ),
        elbow_angle_deg=float(elbow),
        root_excursion_m=float(
            np.linalg.norm(root[:, :2] - root[:1, :2], axis=-1).max()
        ),
        foot_excursion_m=float(
            np.linalg.norm(joints[:, [7, 8], :] - joints[:1, [7, 8], :], axis=-1).max()
        ),
        torso_tilt_rad=float(
            np.arctan2(np.linalg.norm(torso[:, :2], axis=-1), torso[:, 2]).max()
        ),
        heading_span_deg=float(np.rad2deg(np.ptp(heading))),
    )
    distances = np.linalg.norm(right - hold.mean(0), axis=-1)
    distance_changes = np.diff(distances)
    initial_arm = (right[:3] - joints[:3, 17]).mean(0)
    metrics.update(
        initial_arm_elevation_deg=float(
            np.rad2deg(np.arctan2(np.linalg.norm(initial_arm[:2]), -initial_arm[2]))
        ),
        approach_efficiency=float(
            right_displacement
            / max(np.linalg.norm(np.diff(right, axis=0), axis=-1).sum(), 1e-8)
        ),
        target_progress_fraction=float(
            np.maximum(-distance_changes, 0).sum()
            / max(np.abs(distance_changes).sum(), 1e-8)
        ),
        initial_wrist_above_root_m=float((right[:3, 2] - root[:3, 2]).mean()),
        maximum_left_wrist_above_root_m=float((left[:, 2] - root[:, 2]).max()),
        target_horizontal_from_shoulder_m=float(
            np.linalg.norm(
                (
                    joints[-rules.hold_frames :, 21, :2]
                    - joints[-rules.hold_frames :, 17, :2]
                ).mean(0)
            )
        ),
        target_above_root_m=float(
            (right[-rules.hold_frames :, 2] - root[-rules.hold_frames :, 2]).mean()
        ),
    )
    checks = dict(
        initial_arm_already_extended=metrics["initial_arm_elevation_deg"]
        > rules.maximum_initial_arm_elevation_deg,
        indirect_or_repeated_arm_motion=metrics["approach_efficiency"]
        < rules.minimum_approach_efficiency
        or metrics["target_progress_fraction"] < rules.minimum_target_progress_fraction,
        starts_from_existing_target_or_retraction=metrics["initial_wrist_above_root_m"]
        > rules.maximum_initial_wrist_above_root_m,
        raised_noncontrolled_arm=metrics["maximum_left_wrist_above_root_m"]
        > rules.maximum_left_wrist_above_root_m,
        target_is_relaxed_hand_position=metrics["target_horizontal_from_shoulder_m"]
        < rules.minimum_target_horizontal_from_shoulder_m
        and metrics["target_above_root_m"] < rules.minimum_raised_target_above_root_m,
        unstable_target=metrics["hold_span_m"] > rules.maximum_hold_span_m
        or metrics["hold_speed_m_s"] > rules.maximum_hold_speed_m_s,
        insufficient_approach=right_displacement < rules.minimum_wrist_displacement_m,
        arm_not_extended=metrics["arm_extension_m"] < rules.minimum_arm_extension_m
        or elbow < rules.minimum_elbow_angle_deg,
        body_translation=metrics["root_excursion_m"] > rules.maximum_root_excursion_m,
        foot_motion=metrics["foot_excursion_m"] > rules.maximum_foot_excursion_m,
        torso_bending=metrics["torso_tilt_rad"] > rules.maximum_torso_tilt_rad,
        bilateral_or_left_action=left_displacement
        > rules.maximum_left_wrist_displacement_m
        or right_displacement < rules.minimum_right_left_ratio * left_displacement,
        body_turn=metrics["heading_span_deg"] > rules.maximum_heading_span_deg,
    )
    return metrics, [name for name, failed in checks.items() if failed]


@hydra.main(
    version_base="1.3", config_path="../../config", config_name="reach_xyz_repair"
)
def main(config):
    torch.set_num_threads(2)
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "reach_index.json").exists():
        raise ValueError("Completed output exists")
    OmegaConf.save(config, output / "config.yaml", resolve=True)
    frames, sequences, unused, uncertain, categories = load_labels(config)
    candidates = []
    for family, events in frames.items():
        candidates.extend(event for event in events if compatible_event(event))
    for family, events in sequences.items():
        if any(compatible_event(event) for event in frames[family]):
            continue
        # Sequence fallback requires explicitly single-action compatible evidence.
        candidates.extend(
            event
            for event in events
            if event["single_action"] and compatible_event(event)
        )
    base_rows = {
        split: json.loads((Path(config.data) / f"{split}.json").read_text())
        for split in ["train", "val"]
    }
    known = {
        record["family"]: record["motion_source"]
        for rows in base_rows.values()
        for record in rows
        if record.get("motion_source", "").endswith(".npy")
    }
    skeleton = Skeleton(config.skeleton)
    sources = {}
    accepted = []
    rejected = []
    occupied = defaultdict(list)
    provenance = {
        str(Path(config.data) / f"{split}.json"): file_sha256(
            Path(config.data) / f"{split}.json"
        )
        for split in base_rows
    }
    candidates.sort(
        key=lambda event: (
            not event["family"].startswith("MPI_HDM05"),
            event["annotation_level"] != "frame",
            event["family"],
            event["label"].get("start_t", 0),
        )
    )
    for event in candidates:
        family = event["family"]
        if event["split"] not in ["train", "val"]:
            rejected.append(dict(event, reasons=["test_or_unassigned"]))
            continue
        raw = Path(config.amass) / (family + ".npz")
        if not raw.is_file():
            rejected.append(dict(event, reasons=["missing_raw"]))
            continue
        if family not in sources:
            path = Path(
                known.get(
                    family,
                    str(
                        Path(config.project)
                        / "outputs_amass/unified_direct_20261008/data/motions"
                        / (family + ".npy")
                    ),
                )
            )
            if not path.is_file():
                path = output / "converted_motions" / (family + ".npy")
                result = convert_source(
                    (
                        str(
                            Path(config.project) / "work/unified_direct/prepare_data.py"
                        ),
                        (family, str(raw), str(path), config.skeleton),
                    )
                )
                if result["status"] not in ["ok", "cached"]:
                    rejected.append(dict(event, reasons=["conversion_failed"]))
                    continue
            motion = np.load(path)
            with torch.no_grad():
                joints = skeleton(torch.from_numpy(motion)[None])[0].numpy()
            sources[family] = (path, motion, joints)
            provenance[str(path)] = file_sha256(path)
            provenance[str(raw)] = file_sha256(raw)
        path, motion, joints = sources[family]
        label = event["label"]
        start = max(0, int(label.get("start_t", 0) * 20))
        stop = min(len(motion), int(label.get("end_t", event["duration"]) * 20))
        lower = max(0, start - 20)
        # Conflicting frame labels block both approach and hold frames.
        valid = np.ones(len(motion), dtype=bool)
        for other in frames[family]:
            other_label = other["label"]
            if set(other_label.get("act_cat") or []) - CONTEXT:
                valid[
                    max(0, int(other_label["start_t"] * 20)) : min(
                        len(motion), int(np.ceil(other_label["end_t"] * 20))
                    )
                ] = False
        options = []
        failures = Counter()
        for end in range(
            max(lower + config.rules.minimum_frames, start + config.rules.hold_frames),
            stop + 1,
        ):
            for begin in range(
                max(lower, end - config.rules.maximum_frames),
                end - config.rules.minimum_frames + 1,
                2,
            ):
                if not valid[begin:end].all():
                    failures["conflicting_frame_context"] += 1
                    continue
                if any(
                    min(end, prior_end) > max(begin, prior_begin)
                    for prior_begin, prior_end in occupied[family]
                ):
                    continue
                metrics, reasons = crop_metrics(joints[begin:end], config.rules)
                if reasons:
                    failures.update(reasons)
                    continue
                # Prefer well-settled targets and a complete approach over a peak frame.
                score = (
                    metrics["hold_speed_m_s"]
                    + metrics["hold_span_m"]
                    - 0.2 * metrics["right_wrist_displacement_m"]
                )
                options.append((score, begin, end, metrics))
        if not options:
            rejected.append(
                dict(
                    event,
                    reasons=["no_clean_right_approach_and_hold"],
                    failed_windows=dict(failures),
                )
            )
            continue
        score, begin, end, metrics = min(
            options, key=lambda option: (option[0], -(option[2] - option[1]), option[1])
        )
        clip = motion[begin:end].copy()
        with torch.no_grad():
            cropped = skeleton(torch.from_numpy(clip)[None])
            target = target_xyz(cropped)[0].numpy()
        occupied[family].append((begin, end))
        key = (
            "reach_xyz_"
            + hashlib.sha256(f"{family}:{begin}:{end}".encode()).hexdigest()[:20]
        )
        accepted.append(
            dict(
                event,
                key=key,
                task="reach",
                task_id=1,
                caption="A person stands still and reaches with the right hand to a target, then holds the hand there.",
                original_caption=label.get("proc_label") or label.get("raw_label"),
                semantic_revision=REACH_REVISION,
                target_xyz_m=target.tolist(),
                quantity=target.tolist(),
                motion_source=str(path),
                raw_path=str(raw),
                crop_start_frame_20fps=begin,
                crop_end_frame_20fps=end,
                real_frames=len(clip),
                target_frames=len(clip),
                pad_frames=0,
                metrics=metrics,
                source_kind=(
                    "hdm05_babel" if family.startswith("MPI_HDM05") else "babel"
                ),
                boundary_status="kinematic_approach_and_final_hold; not official HDM cut",
                review_status="awaiting_user_review",
                manual_verified=False,
            )
        )
    summary = dict(
        samples={
            split: sum(record["split"] == split for record in accepted)
            for split in ["train", "val"]
        },
        families={
            split: len(
                {record["family"] for record in accepted if record["split"] == split}
            )
            for split in ["train", "val"]
        },
        candidate_events=len(candidates),
        sources=dict(Counter(record["source_kind"] for record in accepted)),
        rejection_reasons=dict(
            Counter(reason for record in rejected for reason in record["reasons"])
        ),
        training_started=False,
        inspect_only=bool(config.inspect_only),
        active_tasks=19,
        retired_task="point",
    )
    save_json(output / "inspection.json", accepted)
    save_json(output / "rejected.json", rejected)
    save_json(output / "summary.json", summary)
    save_json(output / "provenance.json", provenance)
    print(json.dumps(summary, indent=2), flush=True)
    if config.inspect_only:
        return
    encoder = EventTextEncoder(config.project, output)
    (output / "cache").mkdir(exist_ok=True)
    for record in accepted:
        begin, end = record["crop_start_frame_20fps"], record["crop_end_frame_20fps"]
        clip = np.load(record["motion_source"], mmap_mode="r")[begin:end].copy()
        text, local, mask = encoder.make_arrays(
            record["caption"],
            np.arange(begin, end),
            begin / 20,
            end / 20,
            ["action", "right_arm"],
        )
        cache = output / "cache" / f'{record["split"]}_{record["key"]}.npz'
        np.savez(
            cache,
            motion=clip,
            tx=text,
            local=local,
            local_mask=mask,
            quantity=np.asarray(record["target_xyz_m"], dtype=np.float32),
            task=np.int64(1),
        )
        record.update(
            cache=str(cache), cache_sha256=file_sha256(cache), ready_for_training=True
        )
    encoder.save()
    for split, rows in base_rows.items():
        retained = [
            record for record in rows if record["task"] not in ["reach", "point"]
        ]
        combined = retained + [
            record for record in accepted if record["split"] == split
        ]
        save_json(output / f"{split}.json", combined)
    save_json(output / "reach_index.json", accepted)


if __name__ == "__main__":
    main()
