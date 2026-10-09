"""Recover HDM05 walkLeft interiors and mirror into the existing right-only task."""

from collections import Counter
import json
from pathlib import Path

import hydra
import numpy as np
from omegaconf import OmegaConf
from scipy.signal import find_peaks
import torch

from data_processing.event_text import EventTextEncoder
from data_processing.sidestep.rules import check_clip
from data_processing.sidestep.rules import mirror_body
from scripts.prepare_amass20 import convert_source
from shared_motion.training.catalog import HUMAN_HEIGHT
from shared_motion.training.catalog import measure
from shared_motion.training.geometry import Skeleton
from shared_motion.training.model import file_sha256
from shared_motion.training.runner import save_json


def babel_evidence(directory, family):
    evidence = []
    for filename in ["train.json", "val.json", "extra_train.json", "extra_val.json"]:
        for identifier, sequence in json.loads(
            (directory / filename).read_text()
        ).items():
            if not sequence["feat_p"].endswith(family + ".npz"):
                continue
            blocks = (
                (sequence.get("frame_anns") or [])
                if filename.startswith("extra")
                else [sequence.get("frame_ann")]
            )
            for block in blocks:
                if block:
                    evidence.extend(
                        dict(file=filename, sequence_id=identifier, **label)
                        for label in block["labels"]
                    )
    return evidence


@hydra.main(
    version_base="1.3", config_path="../../config", config_name="sidestep_repair"
)
def main(config):
    output = Path(config.output)
    assert (output / "inspection/inventory.json").exists()
    assert not (
        output / "sidestep_index.json"
    ).exists(), "Do not overwrite a repaired dataset"
    for name in ["cache", "native_left_cache", "mirrored_raw", "motions"]:
        (output / name).mkdir(exist_ok=True)
    torch.set_num_threads(2)
    skeleton = Skeleton(config.skeleton)
    scale = HUMAN_HEIGHT / skeleton.height
    encoder = EventTextEncoder(config.project, output)
    inventory = json.loads((output / "inspection/inventory.json").read_text())
    accepted = []
    rejected = []
    sources = {}
    recovered_events = []
    for source in inventory:
        if source["status"] != "inspected":
            rejected.append(dict(source, reasons=["held_out_source_family"]))
            continue
        family = source["family"]
        if len(source["candidates"]) != 1:
            rejected.append(
                dict(source, reasons=["ambiguous_or_missing_walkLeft_phase"])
            )
            continue
        event = source["candidates"][0]
        original_motion = np.load(source["motion"])
        with np.load(source["raw"]) as archive:
            poses = archive["poses"][:, :66].copy()
            translation = archive["trans"].copy()
        mirrored_poses, mirrored_translation = mirror_body(poses, translation)
        restored_poses, restored_translation = mirror_body(
            mirrored_poses, mirrored_translation
        )
        assert np.array_equal(restored_poses, poses) and np.array_equal(
            restored_translation, translation
        )
        mirrored_raw = output / "mirrored_raw" / Path(source["raw"]).name
        np.savez_compressed(
            mirrored_raw,
            poses=mirrored_poses,
            trans=mirrored_translation,
            mocap_framerate=source["raw_fps"],
        )
        mirrored_motion_path = (
            output / "motions" / f'{Path(source["raw"]).stem}_mirror_right.npy'
        )
        conversion = convert_source(
            (
                str(Path(config.project) / "work/unified_direct/prepare_data.py"),
                (family, str(mirrored_raw), str(mirrored_motion_path), config.skeleton),
            )
        )
        assert conversion["status"] in ["ok", "cached"], conversion
        mirrored_motion = np.load(mirrored_motion_path)
        assert original_motion.shape == mirrored_motion.shape
        inspection = np.load(output / "inspection" / f'{Path(source["raw"]).stem}.npz')
        separation = inspection["ankle_separation"]
        closures = (
            find_peaks(
                -separation[event["start"] : event["stop"]],
                prominence=config.rules.closure_prominence_m,
                distance=config.rules.closure_distance_frames,
            )[0]
            + event["start"]
        ).tolist()
        boundaries = sorted(set([event["start"], *closures, event["stop"] - 1]))
        evidence = babel_evidence(Path(config.babel), family)
        nearby = [
            label
            for label in evidence
            if min(label["end_t"], event["end_s"])
            > max(label["start_t"], event["start_s"])
        ]
        recovered = dict(
            family=family,
            split=source["split"],
            semantic_class="walkLeft",
            official_scene="01-01, phase6: sideways left, no crossover",
            boundary_status="supplemented_from_motion_and_script_not_official_cut_mapping",
            native_class_variants_reference=["walkLeft2Steps", "walkLeft3Steps"],
            event=event,
            closure_boundaries_20fps=boundaries,
            babel_corrobating_labels=nearby,
        )
        recovered_events.append(recovered)
        sources[family] = dict(
            source,
            mirrored_raw=str(mirrored_raw),
            mirrored_raw_sha256=file_sha256(mirrored_raw),
            mirrored_motion=str(mirrored_motion_path),
            mirrored_motion_sha256=file_sha256(mirrored_motion_path),
            conversion=conversion,
            mirror_involution_exact=True,
        )
        for start, end_inclusive in zip(boundaries[:-1], boundaries[1:]):
            stop = end_inclusive + 1
            record = dict(
                family=family,
                split=source["split"],
                task="sidestep",
                task_id=5,
                source_kind="hdm05_walkLeft_script_supplement_mirrored",
                source_class="walkLeft",
                native_direction="left",
                training_direction="right",
                mirrored=True,
                annotation_level="script_and_motion_supplement",
                crop_start_frame_20fps=start,
                crop_end_frame_20fps=stop,
                source_event_start_20fps=event["start"],
                source_event_end_20fps=event["stop"],
                raw_source=source["raw"],
                motion_source=str(mirrored_motion_path),
                native_motion_source=source["motion"],
                real_frames=stop - start,
                target_frames=stop - start,
                pad_frames=0,
                caption="side step to the right without crossing the feet",
                boundary_status=recovered["boundary_status"],
            )
            left_clip = original_motion[start:stop].copy()
            right_clip = mirrored_motion[start:stop].copy()
            if (
                not config.rules.minimum_clip_frames
                <= len(right_clip)
                <= config.rules.maximum_clip_frames
            ):
                rejected.append(
                    dict(record, reasons=["cycle_duration_outside_1_to_6_seconds"])
                )
                continue
            with torch.no_grad():
                left_joints = skeleton(torch.from_numpy(left_clip)[None])[0].numpy()
                right_joints = skeleton(torch.from_numpy(right_clip)[None])[0].numpy()
            left_metrics, left_failures = check_clip(left_joints, "left", config.rules)
            right_metrics, right_failures = check_clip(
                right_joints, "right", config.rules
            )
            if left_failures or right_failures:
                rejected.append(
                    dict(
                        record,
                        reasons=left_failures + right_failures,
                        native_metrics=left_metrics,
                        metrics=right_metrics,
                    )
                )
                continue
            with torch.no_grad():
                quantity = float(
                    measure(
                        skeleton,
                        torch.from_numpy(right_clip)[None],
                        torch.tensor([5]),
                        torch.tensor([len(right_clip)]),
                    )[0]
                )
                left_legacy_quantity = float(
                    measure(
                        skeleton,
                        torch.from_numpy(left_clip)[None],
                        torch.tensor([5]),
                        torch.tensor([len(left_clip)]),
                    )[0]
                )
            native_distance = float(
                (left_joints[:, 0, 1] - left_joints[0, 0, 1]).max() * scale
            )
            assert quantity > 0 and native_distance > 0
            identifier = f'sidestep_{Path(source["raw"]).stem}_{start}_{stop}_right'
            cache = output / "cache" / f'{source["split"]}_{identifier}.npz'
            native_cache = (
                output
                / "native_left_cache"
                / f'{source["split"]}_{identifier.replace("_right","_left")}.npz'
            )
            for cache_path, clip, label, distance in [
                (cache, right_clip, record["caption"], quantity),
                (
                    native_cache,
                    left_clip,
                    "side step to the left without crossing the feet",
                    native_distance,
                ),
            ]:
                global_text, local_text, local_mask = encoder.make_arrays(
                    label,
                    np.arange(start, stop),
                    start / 20,
                    stop / 20,
                    ["action", "left_leg", "right_leg"],
                )
                np.savez(
                    cache_path,
                    motion=clip,
                    local=local_text,
                    local_mask=local_mask,
                    tx=global_text,
                    quantity=np.float32(distance),
                    task=np.int64(5 if cache_path == cache else -1),
                )
            accepted.append(
                dict(
                    record,
                    key=identifier,
                    cache=str(cache),
                    cache_sha256=file_sha256(cache),
                    native_left_cache=str(native_cache),
                    native_left_cache_sha256=file_sha256(native_cache),
                    quantity=quantity,
                    native_left_directed_quantity=native_distance,
                    native_left_legacy_task_quantity=left_legacy_quantity,
                    native_metrics=left_metrics,
                    metrics=right_metrics,
                    ready_for_training=True,
                    manual_verified=False,
                    review_status="awaiting_user_review",
                    transformation="Swap22 SMPL body joints, reflect world/rest X using axial-vector sign changes, standard20fps conversion and fixed-skeleton FK; not retimed/scaled",
                    source_fps=source["raw_fps"],
                )
            )
    encoder.save()
    for split in ["train", "val"]:
        prior = json.loads((Path(config.base) / f"{split}.json").read_text())
        save_json(
            output / f"{split}.json",
            [record for record in prior if record["task"] != "sidestep"]
            + [record for record in accepted if record["split"] == split],
        )
    save_json(output / "sidestep_index.json", accepted)
    save_json(
        output / "native_left_index.json",
        [
            dict(
                record,
                cache=record["native_left_cache"],
                cache_sha256=record["native_left_cache_sha256"],
                quantity=record["native_left_directed_quantity"],
                task="sidestep_native_left_reference",
                task_id=None,
                ready_for_training=False,
                training_direction=None,
                purpose="Original left reference only; current task5 measures right travel, so do not feed this directed-left quantity to existing loader",
            )
            for record in accepted
        ],
    )
    save_json(output / "recovered_events.json", recovered_events)
    save_json(output / "sources.json", sources)
    save_json(output / "rejected.json", rejected)
    OmegaConf.save(config, output / "config.yaml", resolve=True)
    audit = dict(
        base=config.base,
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
        rejected_reasons=dict(
            Counter(reason for record in rejected for reason in record["reasons"])
        ),
        native_boundary_status="Official mocapTakes2Cuts mapping HTTP403; only HDM05 scene01-01 noncrossing left phases supplemented from body-local motion, with available BABEL act_cat frame evidence. Not claimed to be official cut rows.",
        other19_tasks_unchanged=True,
        training_direction="right",
        native_left_preserved=True,
        no_training_launched=True,
        limitations=[
            "Mirrored SMPL poses are re-FKed on the original asymmetric skeleton, so the result is not an exact pixelwise reflection.",
            "Adjacent clips share at most one closure frame and inherit original source-family split.",
            "One full open-close cycle is cropped; original official2/3-step cut boundaries are not claimed.",
        ],
    )
    save_json(output / "audit.json", audit)
    report = Path(config.report)
    report.mkdir(parents=True, exist_ok=True)
    save_json(report / "audit.json", audit)
    print(json.dumps(audit), flush=True)


if __name__ == "__main__":
    main()
