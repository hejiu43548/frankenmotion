"""Use BABEL categories and actual right-forward leg excursions; retain all audit evidence."""

from collections import Counter
import json
from pathlib import Path
import re

import hydra
import numpy as np
from omegaconf import OmegaConf
from scipy.signal import find_peaks
import torch

from data_processing.event_text import EventTextEncoder
from scripts.prepare_amass20 import convert_source
from shared_motion.training.catalog import measure
from shared_motion.training.geometry import Skeleton
from shared_motion.training.model import file_sha256
from shared_motion.training.runner import save_json

COMPATIBLE = {
    "kick",
    "leg movements",
    "forward movement",
    "martial art",
    "sports move",
    "exercise/training",
    "interact with/use object",
}
CONTEXT = COMPATIBLE | {
    "stand",
    "transition",
    "move body part",
    "body part movement",
    "raising body part",
    "lowering body part",
}
WRONG_STYLE = re.compile(
    r"\bleft\b|\bside\w*\b|\bback\w*\b|\bround\w*\b|\bspin\w*\b|\bboth\b|\balternat\w*\b",
    re.I,
)


def semantic_failures(label):
    failures = []
    if "kick" not in (label.get("act_cat") or []):
        failures.append("no_kick_category")
    if set(label.get("act_cat") or []) - COMPATIBLE:
        failures.append("conflicting_category")
    if WRONG_STYLE.search(label.get("proc_label") or label.get("raw_label") or ""):
        failures.append("left_side_back_round_or_mixed_kick")
    return failures


def leg_coordinates(joints, ankle_index=8):
    lateral = joints[:, 1, :2] - joints[:, 2, :2]
    lateral /= np.maximum(np.linalg.norm(lateral, axis=-1, keepdims=True), 1e-8)
    forward = np.stack([lateral[:, 1], -lateral[:, 0]], axis=-1)
    relative = joints[:, ankle_index] - joints[:, 0]
    return (relative[:, :2] * forward).sum(-1), (relative[:, :2] * lateral).sum(-1)


def physical_check(joints, peak_frame, rules):
    right_forward, right_lateral = leg_coordinates(joints)
    left_forward, _ = leg_coordinates(joints, 7)
    ankle_height = joints[:, 8, 2]
    root = joints[:, 0]
    lateral_hips = joints[:, 1, :2] - joints[:, 2, :2]
    heading = np.unwrap(np.arctan2(lateral_hips[:, 1], lateral_hips[:, 0]))
    target_window = slice(max(0, peak_frame - 3), min(len(joints), peak_frame + 4))
    preceding = slice(max(0, peak_frame - 18), peak_frame + 1)
    forward_excursion = float(
        right_forward[target_window].max() - right_forward[preceding].min()
    )
    left_peaks = find_peaks(left_forward, prominence=0.18, height=0.20, distance=10)[0]
    left_kicks = [
        int(frame)
        for frame in left_peaks
        if joints[frame, 7, 2] - joints[:, 7, 2].min() > 0.10
    ]
    metrics = dict(
        forward_excursion_m=forward_excursion,
        ankle_lift_m=float(ankle_height[target_window].max() - ankle_height.min()),
        lateral_excursion_m=float(
            np.max(np.abs(right_lateral[target_window] - right_lateral[0]))
        ),
        root_excursion_m=float(
            np.linalg.norm(root[:, :2] - root[:1, :2], axis=-1).max()
        ),
        support_foot_excursion_m=float(
            np.linalg.norm(
                joints[:, [7, 10], :2] - joints[:1, [7, 10], :2], axis=-1
            ).max()
        ),
        heading_change_rad=float(np.ptp(heading)),
        left_kick_frames=left_kicks,
        source_metric_peak_frame=int(
            10 + np.argmax((joints[:, 8, 0] - root[:, 0])[10:51])
        ),
    )
    failures = []
    if forward_excursion < rules.minimum_forward_excursion_m:
        failures.append("insufficient_forward_kick")
    if metrics["ankle_lift_m"] < rules.minimum_ankle_lift_m:
        failures.append("no_raised_kicking_ankle")
    if metrics["lateral_excursion_m"] > max(
        0.12, forward_excursion * rules.maximum_lateral_ratio
    ):
        failures.append("lateral_or_round_kick")
    for metric, maximum in [
        ("root_excursion_m", rules.maximum_root_excursion_m),
        ("support_foot_excursion_m", rules.maximum_support_foot_excursion_m),
        ("heading_change_rad", rules.maximum_heading_change_rad),
    ]:
        if metrics[metric] > maximum:
            failures.append(metric + "_above_limit")
    if left_kicks:
        failures.append("left_kick_inside_crop")
    if abs(metrics["source_metric_peak_frame"] - peak_frame) > 5:
        failures.append("command_peak_not_selected_event")
    # The first frame is the legacy command's reference; avoid starting with an extended leg.
    if right_forward[0] > right_forward[target_window].max() - 0.7 * forward_excursion:
        failures.append("crop_starts_mid_kick")
    return metrics, failures


@hydra.main(version_base="1.3", config_path="../../config", config_name="kick_repair")
def main(config):
    project = Path(config.project)
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=False)
    (output / "cache").mkdir()
    report_directory = Path(config.report)
    report_directory.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output / "config.yaml", resolve=True)
    torch.set_num_threads(2)
    annotations_root = project / "datasets/annotations/frankenstein-dataset/annotations"
    annotations = json.loads((annotations_root / "annotations.json").read_text())
    split_by_family = {}
    for split in ["train", "val", "test"]:
        for annotation_key in (
            (annotations_root / "splits" / f"{split}.txt").read_text().split()
        ):
            family = annotations[annotation_key]["path"]
            assert split_by_family.setdefault(family, split) == split
    raw_index_path = (
        project
        / "outputs_amass/stage2_manifests_20261008/annotated_source_raw_audit.json"
    )
    raw_index = json.loads(raw_index_path.read_text())
    skeleton_path = (
        project
        / "outputs_amass/transfer_charlie_20261008/snapshot/outputs_amass/franken_eleven_20261003/skeleton.npz"
    )
    skeleton = Skeleton(skeleton_path)
    candidates = []
    rejected = []
    provenance = {}
    for filename in ["train.json", "val.json", "extra_train.json", "extra_val.json"]:
        annotation_path = Path(config.babel) / filename
        provenance[str(annotation_path)] = file_sha256(annotation_path)
        for sequence_id, sequence in json.loads(annotation_path.read_text()).items():
            family = str(Path(*Path(sequence["feat_p"]).parts[1:]).with_suffix(""))
            frame_blocks = (
                (sequence.get("frame_anns") or [])
                if filename.startswith("extra")
                else [sequence.get("frame_ann")]
            )
            blocks = [(block, "frame") for block in frame_blocks if block]
            if not blocks:
                sequence_blocks = (
                    (sequence.get("seq_anns") or [])
                    if filename.startswith("extra")
                    else [sequence.get("seq_ann")]
                )
                for block in sequence_blocks:
                    if (
                        block
                        and block.get("mul_act") is False
                        and block.get("labels")
                        and all(
                            not semantic_failures(label) for label in block["labels"]
                        )
                    ):
                        blocks.append((block, "single_action_sequence"))
            for block, annotation_level in blocks:
                for label in block["labels"]:
                    if "kick" not in (label.get("act_cat") or []):
                        continue
                    event_label = dict(label)
                    if annotation_level == "single_action_sequence":
                        event_label.update(start_t=0, end_t=sequence["dur"])
                    record = dict(
                        family=family,
                        split=split_by_family.get(family),
                        babel_file=filename,
                        babel_sid=sequence_id,
                        babel_lid=block.get("babel_lid"),
                        seg_id=label["seg_id"],
                        caption=label["proc_label"],
                        source_label=event_label,
                        annotation_level=annotation_level,
                        duration_s=sequence["dur"],
                        source_kind=(
                            "hdm05_babel_kick"
                            if family.startswith("MPI_HDM05/") and "_03-02_" in family
                            else "babel_kick"
                        ),
                    )
                    failures = semantic_failures(label)
                    if record["split"] not in ["train", "val"]:
                        failures.append("test_or_unassigned_source_family")
                    if raw_index.get(family, {}).get("status") != "raw_header_valid":
                        failures.append("missing_verified_raw_source")
                    if failures:
                        rejected.append(dict(record, reasons=failures))
                    else:
                        candidates.append(
                            (
                                record,
                                block["labels"] if annotation_level == "frame" else [],
                            )
                        )
    candidates.sort(
        key=lambda item: (
            item[0]["source_kind"] != "hdm05_babel_kick",
            item[0]["annotation_level"] != "frame",
            item[0]["family"],
            item[0]["source_label"]["start_t"],
        )
    )
    loaded_sources = {}
    source_hashes = {}
    accepted = []
    seen_windows = {}
    encoder = EventTextEncoder(project, output)
    for record, context_labels in candidates:
        family = record["family"]
        if family not in loaded_sources:
            motion_path = (
                project
                / "outputs_amass/unified_direct_20261008/data/motions"
                / (family + ".npy")
            )
            if not motion_path.exists():
                motion_path = output / "motions" / (family + ".npy")
                result = convert_source(
                    (
                        str(project / "work/unified_direct/prepare_data.py"),
                        (
                            family,
                            raw_index[family]["raw_path"],
                            str(motion_path),
                            str(skeleton_path),
                        ),
                    )
                )
                assert result["status"] in ["ok", "cached"], result
            motion = np.load(motion_path)
            with torch.no_grad():
                joints = skeleton(torch.from_numpy(motion)[None])[0].numpy()
            loaded_sources[family] = motion_path, motion, joints
            source_hashes[family] = dict(
                motion=str(motion_path),
                motion_sha256=file_sha256(motion_path),
                raw=raw_index[family]["raw_path"],
                raw_sha256=file_sha256(raw_index[family]["raw_path"]),
            )
        motion_path, motion, joints = loaded_sources[family]
        if abs(len(motion) / 20 - record["duration_s"]) > 0.15:
            rejected.append(dict(record, reasons=["source_duration_mismatch"]))
            continue
        forward_reach, _ = leg_coordinates(joints)
        peaks = find_peaks(
            forward_reach,
            height=config.rules.minimum_peak_reach_m,
            prominence=config.rules.minimum_forward_excursion_m,
            distance=10,
        )[0]
        peaks = [
            int(frame)
            for frame in peaks
            if record["source_label"]["start_t"]
            <= frame / 20
            < record["source_label"]["end_t"]
        ]
        if not peaks:
            rejected.append(dict(record, reasons=["no_right_forward_kick_peak"]))
        for source_peak in peaks:
            attempts = []
            selected = None
            for frames in config.rules.crop_frames:
                for peak_frame in config.rules.peak_frames:
                    start = source_peak - peak_frame
                    stop = start + frames
                    if start < 0 or stop > len(motion):
                        continue
                    clip = motion[start:stop].copy()
                    with torch.no_grad():
                        crop_joints = skeleton(torch.from_numpy(clip)[None])[0].numpy()
                    metrics, failures = physical_check(
                        crop_joints, peak_frame, config.rules
                    )
                    for other_label in context_labels:
                        overlap = max(
                            0,
                            min(stop / 20, other_label["end_t"])
                            - max(start / 20, other_label["start_t"]),
                        )
                        if other_label["seg_id"] == record["seg_id"] or overlap < 0.1:
                            continue
                        categories = set(other_label.get("act_cat") or [])
                        if categories - CONTEXT or (
                            "kick" in categories and semantic_failures(other_label)
                        ):
                            failures.append("conflicting_context_action")
                            break
                    if failures:
                        attempts.append(
                            dict(
                                start=start,
                                stop=stop,
                                peak_frame=peak_frame,
                                metrics=metrics,
                                reasons=failures,
                            )
                        )
                        continue
                    selected = start, stop, peak_frame, clip, metrics
                    break
                if selected is not None:
                    break
            if selected is None:
                rejected.append(
                    dict(
                        record,
                        source_peak_frame_20fps=source_peak,
                        reasons=sorted(
                            {
                                reason
                                for attempt in attempts
                                for reason in attempt["reasons"]
                            }
                        )
                        or ["no_unpadded_context"],
                        crop_attempts=attempts,
                    )
                )
                continue
            start, stop, peak_frame, clip, metrics = selected
            previous = seen_windows.setdefault(family, [])
            if any(
                max(0, min(stop, prior_stop) - max(start, prior_start))
                / min(stop - start, prior_stop - prior_start)
                >= 0.5
                for prior_start, prior_stop in previous
            ):
                rejected.append(
                    dict(
                        record,
                        source_peak_frame_20fps=source_peak,
                        reasons=["overlapping_source_event"],
                    )
                )
                continue
            with torch.no_grad():
                quantity = float(
                    measure(
                        skeleton,
                        torch.from_numpy(clip)[None],
                        torch.tensor([7]),
                        torch.tensor([len(clip)]),
                    )[0]
                )
            assert np.isfinite(quantity) and np.isfinite(clip).all()
            identifier = f"kick_{record['babel_sid']}_{start}_{stop}"
            cache = output / "cache" / f"{record['split']}_{identifier}.npz"
            global_text, local_text, local_mask = encoder.make_arrays(
                record["caption"],
                np.arange(start, stop),
                record["source_label"]["start_t"],
                record["source_label"]["end_t"],
                ["action", "right_leg"],
            )
            np.savez(
                cache,
                motion=clip,
                local=local_text,
                local_mask=local_mask,
                tx=global_text,
                quantity=np.float32(quantity),
                task=np.int64(7),
            )
            accepted.append(
                dict(
                    record,
                    key=identifier,
                    task="kick",
                    task_id=7,
                    cache=str(cache),
                    cache_sha256=file_sha256(cache),
                    motion_source=str(motion_path),
                    crop_start_frame_20fps=start,
                    crop_end_frame_20fps=stop,
                    source_peak_frame_20fps=source_peak,
                    local_peak_frame=peak_frame,
                    quantity=quantity,
                    real_frames=len(clip),
                    target_frames=len(clip),
                    pad_frames=0,
                    metrics=metrics,
                    ready_for_training=True,
                    manual_verified=False,
                    review_status="awaiting_user_review",
                )
            )
            previous.append((start, stop))
    encoder.save()
    for split in ["train", "val"]:
        manifest_path = Path(config.base) / f"{split}.json"
        original = json.loads(manifest_path.read_text())
        provenance[str(manifest_path)] = file_sha256(manifest_path)
        save_json(
            output / f"{split}.json",
            [record for record in original if record["task"] != "kick"]
            + [record for record in accepted if record["split"] == split],
        )
    save_json(output / "kick_index.json", accepted)
    save_json(output / "rejected.json", rejected)
    save_json(output / "sources.json", source_hashes)
    for source_path in [
        Path(__file__),
        Path(__file__).parents[1] / "event_text.py",
        raw_index_path,
        skeleton_path,
        annotations_root / "annotations.json",
    ]:
        provenance[str(source_path)] = file_sha256(source_path)
    save_json(output / "provenance.json", provenance)
    summary = dict(
        status="awaiting_user_review",
        base=config.base,
        counts=dict(Counter(record["split"] for record in accepted)),
        by_source=dict(
            Counter(
                record["split"]
                + ":"
                + record["source_kind"]
                + ":"
                + record["annotation_level"]
                for record in accepted
            )
        ),
        families={
            split: len(
                {record["family"] for record in accepted if record["split"] == split}
            )
            for split in ["train", "val"]
        },
        rejection_counts=dict(
            Counter(reason for record in rejected for reason in record["reasons"])
        ),
        other19_tasks_unchanged=True,
        training_started=False,
        scope="Right forward kick; BABEL act_cat with frame events or explicitly single-action sequences. No regex-only admission, no generated motion or time scaling.",
    )
    save_json(output / "audit.json", summary)
    save_json(report_directory / "audit.json", summary)
    OmegaConf.save(config, report_directory / "config.yaml", resolve=True)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
