#!/usr/bin/env python3
"""Append uncapped real-motion records for main's nine added task classes."""

from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import re
import sys

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared_motion.adapter.catalog import NEW
from shared_motion.training.catalog import ACTIVE_TASK_NAMES, TASK_NAMES, measure
from shared_motion.training.geometry import Skeleton
from shared_motion.training.model import file_sha256
from shared_motion.training.runner import save_json


def convert_source(job):
    module_path, arguments = job
    sys.path.insert(0, str(Path(module_path).parent))
    specification = importlib.util.spec_from_file_location(
        "amass_source_conversion", module_path
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    return module.convert(arguments)


def event_windows(annotation, pattern, frames):
    lower = max(0, int(annotation["start"] * 20))
    upper = min(frames, int(annotation["end"] * 20))
    intervals = []
    for segment in annotation["annotations"]:
        if segment["bodypart"] != "sequence_caption" and re.search(
            pattern, segment["text"], re.I
        ):
            start = max(lower, int(segment["start"] * 20))
            stop = min(upper, int(segment["end"] * 20))
            if stop > start:
                intervals.append((start, stop))
    source = (
        "timed matching body/action annotation"
        if intervals
        else "global-caption match; no timed matching label"
    )
    if not intervals:
        intervals = [(lower, upper)]
    merged = []
    for start, stop in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(stop, merged[-1][1]))
        else:
            merged.append((start, stop))
    windows = []
    for start, stop in merged:
        if stop - start < 40:
            midpoint = (start + stop) // 2
            start = max(lower, midpoint - 20)
            stop = min(upper, start + 40)
            start = max(lower, stop - 40)
        if stop - start < 40:
            continue
        boundaries = np.linspace(
            start, stop, int(np.ceil((stop - start) / 120)) + 1, dtype=int
        )
        windows.extend(
            (int(begin), int(end), source)
            for begin, end in zip(boundaries[:-1], boundaries[1:])
        )
    return windows


def load_running_records(directory, split):
    """Do not regenerate these two tasks from caption regular expressions."""
    if not directory:
        raise ValueError("Set running_data to a verified jog_march repair output")
    path = Path(directory) / f"{split}.json"
    rows = [
        record
        for record in json.loads(path.read_text())
        if record["task"] in ["jog", "march"]
    ]
    if {record["task"] for record in rows} != {"jog", "march"}:
        raise ValueError(f"Missing repaired jog/march records in {path}")
    revisions = dict(jog="straight_running_v2", march="stationary_running_v1")
    for record in rows:
        if (
            record.get("split") != split
            or record.get("semantic_revision") != revisions[record["task"]]
        ):
            raise ValueError("Legacy regex-only running data is not admissible")
        cache = Path(record["cache"])
        if (
            not cache.is_absolute()
            or not cache.is_file()
            or file_sha256(cache) != record.get("cache_sha256")
        ):
            raise ValueError(f"Invalid repaired running cache: {cache}")
    return rows


def load_reach_records(directory, split):
    from shared_motion.training.reach import REACH_REVISION

    if not directory:
        raise ValueError("Set reach_data to a verified XYZ reach repair")
    rows = [
        record
        for record in json.loads((Path(directory) / f"{split}.json").read_text())
        if record["task"] == "reach"
    ]
    if not rows:
        raise ValueError("Missing XYZ reach records")
    for record in rows:
        if (
            record.get("semantic_revision") != REACH_REVISION
            or record.get("split") != split
            or len(record.get("target_xyz_m", [])) != 3
        ):
            raise ValueError("Legacy scalar reach data rejected")
        if file_sha256(record["cache"]) != record.get("cache_sha256"):
            raise ValueError("Reach cache hash mismatch")
    return rows


@hydra.main(version_base="1.3", config_path="../config", config_name="prepare_amass20")
def main(config):
    running = {
        split: load_running_records(config.get("running_data"), split)
        for split in ["train", "val"]
    }
    reaching = {
        split: load_reach_records(config.get("reach_data"), split)
        for split in ["train", "val"]
    }
    output = Path(config.output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    OmegaConf.save(config, output / "config.yaml", resolve=True)
    annotation_root = Path(config.annotations)
    annotations = json.loads((annotation_root / "annotations.json").read_text())
    raw_index = json.loads(Path(config.raw_index).read_text())
    split_ids = {
        split: (annotation_root / "splits" / f"{split}.txt").read_text().split()
        for split in ["train", "val", "test"]
    }
    families = {
        split: {annotations[key]["path"] for key in identifiers}
        for split, identifiers in split_ids.items()
    }
    if any(
        families[first] & families[second]
        for first, second in [("train", "val"), ("train", "test"), ("val", "test")]
    ):
        raise ValueError("Official source-family split overlap")
    candidates = []
    rejected = []
    sources = {}
    for split in ["train", "val"]:
        for key in split_ids[split]:
            annotation = annotations[key]
            for task, definition in NEW.items():
                if task in ["jog", "march", "point"]:
                    continue
                if not re.search(
                    definition["pattern"], annotation["caption_label"], re.I
                ):
                    continue
                family = annotation["path"]
                source = raw_index.get(family, {})
                if (
                    source.get("status") != "raw_header_valid"
                    or "humanact12" in family.lower()
                ):
                    rejected.append(
                        dict(
                            split=split,
                            key=key,
                            task=task,
                            reason="raw source unavailable/invalid",
                        )
                    )
                    continue
                candidates.append(dict(split=split, key=key, task=task, family=family))
                sources[family] = source["raw_path"]
    destinations = {}
    jobs = []
    for family, raw_path in sources.items():
        existing = Path(config.existing_motions) / (family + ".npy")
        destination = (
            existing if existing.exists() else output / "motions" / (family + ".npy")
        )
        destinations[family] = destination
        if not existing.exists():
            jobs.append(
                (
                    config.conversion_module,
                    (family, raw_path, str(destination), config.skeleton),
                )
            )
    conversion = []
    with ProcessPoolExecutor(config.workers) as executor:
        for result in executor.map(convert_source, jobs, chunksize=4):
            conversion.append(result)
            if len(conversion) % 50 == 0:
                save_json(
                    output / "status.json",
                    dict(state="converting", done=len(conversion), total=len(jobs)),
                )
    failed = {
        record["family"]: record["reason"]
        for record in conversion
        if record["status"] == "rejected"
    }
    save_json(output / "conversion.json", conversion)
    embeddings_root = Path(config.embeddings)
    embeddings = np.load(embeddings_root / "clip.npy", mmap_mode="r")
    embedding_index = json.loads((embeddings_root / "clip_index.json").read_text())
    slices = np.load(embeddings_root / "clip_slice.npy")
    with np.load(config.pca) as archive:
        pca_mean = archive["mean"].copy()
        components = archive["components"].copy()
    parts = json.loads(Path(config.official_config).read_text())["data"][
        "text_encoder"
    ]["body_part_order"]
    reduced = {}

    def embedding(label):
        if label not in embedding_index:
            raise ValueError(f"Missing cached text: {label}")
        start, stop = slices[embedding_index[label]]
        value = np.asarray(embeddings[start:stop], dtype=np.float32)
        if value.shape != (1, 512):
            raise ValueError("Expected one global CLIP vector")
        if label not in reduced:
            reduced[label] = (value[0] - pca_mean) @ components.T
        return value[0], reduced[label]

    skeleton = Skeleton(config.skeleton)
    output.joinpath("cache").mkdir()
    accepted = []
    identities = set()
    for record in candidates:
        try:
            if record["family"] in failed:
                raise ValueError(failed[record["family"]])
            annotation = annotations[record["key"]]
            source_path = destinations[record["family"]]
            full = np.load(source_path, mmap_mode="r")
            windows = event_windows(
                annotation, NEW[record["task"]]["pattern"], len(full)
            )
            if not windows:
                raise ValueError("No real window of at least2 seconds")
            for start, stop, evidence in windows:
                identity = (record["task"], record["family"], start, stop)
                if identity in identities:
                    continue
                motion = np.asarray(full[start:stop]).copy()
                frames = len(motion)
                local = np.zeros((frames, 408), np.float32)
                mask = np.zeros((frames, 408), bool)
                times = np.arange(start, stop) / 20
                for part_index, part in enumerate(parts):
                    for frame_index, time in enumerate(times):
                        label = next(
                            (
                                segment["text"]
                                for segment in annotation["annotations"]
                                if segment["bodypart"] == part
                                and segment["start"] <= time < segment["end"]
                            ),
                            "unknown",
                        )
                        if label != "unknown":
                            local[
                                frame_index, part_index * 51 : (part_index + 1) * 51
                            ] = embedding(label)[1]
                            mask[
                                frame_index, part_index * 51 : (part_index + 1) * 51
                            ] = True
                text = embedding(annotation["caption_label"])[0]
                task_index = TASK_NAMES.index(record["task"])
                with torch.no_grad():
                    quantity = float(
                        measure(
                            skeleton,
                            torch.from_numpy(motion)[None],
                            torch.tensor([task_index]),
                            torch.tensor([frames]),
                        )[0]
                    )
                if not np.isfinite(quantity) or not np.isfinite(motion).all():
                    raise ValueError("Nonfinite motion/quantity")
                identifier = hashlib.sha256(repr(identity).encode()).hexdigest()[:20]
                cache = (
                    output
                    / "cache"
                    / f"{record['split']}_{record['task']}_{identifier}.npz"
                )
                np.savez(
                    cache,
                    motion=motion,
                    local=local,
                    local_mask=mask,
                    tx=text,
                    quantity=np.float32(quantity),
                    task=np.int64(task_index),
                )
                accepted.append(
                    dict(
                        record,
                        key=f"{record['key']}_{start}_{stop}",
                        annotation_key=record["key"],
                        task_id=task_index,
                        cache=str(cache),
                        cache_sha256=file_sha256(cache),
                        quantity=quantity,
                        caption=annotation["caption_label"],
                        crop_start_frame_20fps=start,
                        crop_end_frame_20fps=stop,
                        target_frames=frames,
                        real_frames=frames,
                        pad_frames=0,
                        motion_source=str(source_path),
                        semantic_event_verification=evidence,
                        manual_verified=False,
                    )
                )
                identities.add(identity)
        except Exception as error:
            rejected.append(dict(record, reason=repr(error)))
        if (len(accepted) + len(rejected)) % 50 == 0:
            save_json(
                output / "status.json",
                dict(state="caching", accepted=len(accepted), rejected=len(rejected)),
            )
    totals = {}
    for split in ["train", "val"]:
        original = [
            record
            for record in json.loads(Path(config[f"base_{split}"]).read_text())
            if record["task"] not in ["reach", "point"]
        ]
        # Preserve existing11 classes exactly, except resolving their relative cache paths.
        for record in original:
            cache = Path(record["cache"])
            record["cache"] = str(
                cache if cache.is_absolute() else Path(config.cache_root) / cache
            )
        additions = (
            [record for record in accepted if record["split"] == split]
            + running[split]
            + reaching[split]
        )
        rows = original + additions
        counts = Counter(record["task"] for record in rows)
        if set(counts) != set(ACTIVE_TASK_NAMES):
            raise ValueError(
                f"Missing tasks in {split}: {set(ACTIVE_TASK_NAMES)-set(counts)}"
            )
        save_json(output / f"{split}.json", rows)
        totals[split] = dict(counts)
    all_new = (
        accepted
        + running["train"]
        + running["val"]
        + reaching["train"]
        + reaching["val"]
    )
    save_json(output / "new9_index.json", all_new)
    save_json(
        output / "running_provenance.json",
        {
            "directory": str(config.running_data),
            "manifest_sha256": {
                split: file_sha256(Path(config.running_data) / f"{split}.json")
                for split in running
            },
            "policy": "Imported category and gait verified records; no regex admission for jog or march",
        },
    )
    save_json(output / "rejected.json", rejected)
    save_json(
        output / "source_hashes.json",
        {
            family: dict(
                raw_path=sources[family],
                motion=str(path),
                motion_sha256=file_sha256(path),
            )
            for family, path in destinations.items()
            if path.exists()
        },
    )
    save_json(
        output / "status.json",
        dict(
            state="complete",
            counts=totals,
            accepted_new9=len(all_new),
            rejected=len(rejected),
            sample_cap=None,
            split_family_isolation=True,
            pca_sha256=file_sha256(config.pca),
            annotations_sha256=file_sha256(annotation_root / "annotations.json"),
            conversion_source_sha256=file_sha256(config.conversion_module),
            source_sha256=file_sha256(__file__),
            limitations=[
                "Six added tasks use legacy selection; jog/march require repaired gait records; point is merged into XYZ reach.",
                "Global captions can describe a broader sequence; no generated templates.",
                "Running validation source counts and limitations are inherited from running_data; windows are not independent sources.",
            ],
        ),
    )
    print(json.dumps(totals, indent=2), flush=True)


if __name__ == "__main__":
    main()
