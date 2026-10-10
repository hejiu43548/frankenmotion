"""Filter/re-crop vetted jog events to straight running, preserving march exactly."""

from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil

import hydra
import numpy as np
from omegaconf import OmegaConf
import torch

from data_processing.jog_march.rules import classify_clip
from data_processing.jog_march.straight_rules import straight_metrics
from shared_motion.training.catalog import TASK_NAMES
from shared_motion.training.catalog import measure
from shared_motion.training.geometry import Skeleton
from shared_motion.training.model import file_sha256
from shared_motion.training.runner import save_json


def select_straight_windows(joints, gait_rules, path_rules):
    """Longest valid windows first, then remaining nonoverlapping intervals."""
    selected = []
    rejected = Counter()
    for length in range(len(joints), gait_rules.minimum_frames - 1, -1):
        for begin in range(len(joints) - length + 1):
            end = begin + length
            if any(min(end, stop) > max(begin, start) for start, stop in selected):
                continue
            clip = joints[begin:end]
            metrics, failures = straight_metrics(clip, path_rules)
            if failures:
                rejected.update(failures)
                continue
            task, metrics, failures = classify_clip(clip, gait_rules)
            if task != "jog" or failures:
                rejected.update(failures or ["not_jog"])
                continue
            selected.append((begin, end))
    return sorted(selected), dict(rejected)


@hydra.main(
    version_base="1.3", config_path="../../config", config_name="jog_straight_repair"
)
def main(config):
    torch.set_num_threads(2)
    base = Path(config.base)
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "running_index.json").exists():
        raise ValueError("Completed output exists; choose a new directory")
    OmegaConf.save(config, output / "config.yaml", resolve=True)
    skeleton = Skeleton(config.skeleton)
    old_rows = json.loads((base / "running_index.json").read_text())
    accepted = []
    audit = []
    provenance = {
        str(base / "running_index.json"): file_sha256(base / "running_index.json")
    }
    repository = Path(__file__).resolve().parents[2]
    for path in list(Path(__file__).parent.glob("*.py")) + [
        repository / "config/jog_straight_repair.yaml",
        repository / "config/jog_march_repair.yaml",
        repository / "shared_motion/training/catalog.py",
        Path(config.skeleton),
    ]:
        provenance[str(path)] = file_sha256(path)
    for record in old_rows:
        provenance[record["cache"]] = file_sha256(record["cache"])
        assert provenance[record["cache"]] == record["cache_sha256"]
        if record["task"] == "march":
            accepted.append(record)
            continue
        with np.load(record["cache"]) as cache:
            motion = cache["motion"].copy()
        with torch.no_grad():
            joints = skeleton(torch.from_numpy(motion)[None])[0].numpy()
        metrics, failures = straight_metrics(joints, config.straight)
        windows, rejected = select_straight_windows(
            joints, config.rules, config.straight
        )
        audit.append(
            dict(
                key=record["key"],
                family=record["family"],
                split=record["split"],
                original_metrics=metrics,
                original_failures=failures,
                accepted_relative_windows=windows,
                rejected_window_criteria=rejected,
            )
        )
        for begin, end in windows:
            clip = motion[begin:end].copy()
            with torch.no_grad():
                positions = skeleton(torch.from_numpy(clip)[None])[0].numpy()
                quantity = float(
                    measure(
                        skeleton,
                        torch.from_numpy(clip)[None],
                        torch.tensor([TASK_NAMES.index("jog")]),
                        torch.tensor([len(clip)]),
                    )[0]
                )
            task, gait_metrics, gait_failures = classify_clip(positions, config.rules)
            path_metrics, path_failures = straight_metrics(positions, config.straight)
            assert task == "jog" and not gait_failures and not path_failures
            lower = record["crop_start_frame_20fps"] + begin
            upper = record["crop_start_frame_20fps"] + end
            key = (
                "jog_straight_"
                + hashlib.sha256(
                    f'{record["family"]}:{lower}:{upper}'.encode()
                ).hexdigest()[:20]
            )
            accepted.append(
                dict(
                    record,
                    key=key,
                    parent_key=record["key"],
                    parent_cache=record["cache"],
                    crop_start_frame_20fps=lower,
                    crop_end_frame_20fps=upper,
                    real_frames=len(clip),
                    target_frames=len(clip),
                    quantity=quantity,
                    semantic_revision="straight_running_v2",
                    metrics=gait_metrics,
                    straight_metrics=path_metrics,
                    boundary_status="straight_running_subcrop_of_vetted_BABEL_running_event",
                    review_status="awaiting_user_review",
                )
            )
    summary = dict(
        tasks={},
        training_started=False,
        inspect_only=bool(config.inspect_only),
        scope="Only jog changed; march and other18 records preserved",
        original_jog_clips=len(audit),
        original_full_clips_straight=sum(
            not entry["original_failures"] for entry in audit
        ),
        original_jog_clips_with_straight_subcrop=sum(
            bool(entry["accepted_relative_windows"]) for entry in audit
        ),
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
    save_json(output / "inspection.json", accepted)
    save_json(output / "straight_audit.json", audit)
    save_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)
    if config.inspect_only:
        return
    (output / "cache").mkdir(exist_ok=True)
    for record in accepted:
        if record["task"] == "march":
            continue
        parent = next(
            original for original in old_rows if original["key"] == record["parent_key"]
        )
        begin = record["crop_start_frame_20fps"] - parent["crop_start_frame_20fps"]
        end = record["crop_end_frame_20fps"] - parent["crop_start_frame_20fps"]
        with np.load(record["parent_cache"]) as cache:
            arrays = {name: cache[name].copy() for name in cache.files}
        for name in ["motion", "local", "local_mask"]:
            arrays[name] = arrays[name][begin:end].copy()
        arrays["quantity"] = np.float32(record["quantity"])
        cache_path = output / "cache" / f'{record["split"]}_{record["key"]}.npz'
        np.savez(cache_path, **arrays)
        record.update(cache=str(cache_path), cache_sha256=file_sha256(cache_path))
        provenance[record["motion_source"]] = file_sha256(record["motion_source"])
    for split in ["train", "val"]:
        path = base / f"{split}.json"
        before = json.loads(path.read_text())
        provenance[str(path)] = file_sha256(path)
        after = [
            record for record in before if record["task"] not in ["jog", "march"]
        ] + [record for record in accepted if record["split"] == split]
        assert [record for record in before if record["task"] != "jog"] == [
            record for record in after if record["task"] != "jog"
        ]
        save_json(output / f"{split}.json", after)
    for filename in [
        "new_event_embeddings.json",
        "new_event_embeddings.npz",
        "uncertain_time_labels.json",
    ]:
        shutil.copyfile(base / filename, output / filename)
        provenance[str(base / filename)] = file_sha256(base / filename)
    save_json(output / "running_index.json", accepted)
    save_json(output / "provenance.json", provenance)


if __name__ == "__main__":
    main()
