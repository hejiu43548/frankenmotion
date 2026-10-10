"""Full crop, class, command, conditioning and split verification for both tasks."""

from collections import Counter
from collections import defaultdict
import json
from pathlib import Path
import shutil

import hydra
import numpy as np
import torch

from data_processing.jog_march.rules import classify_clip
from data_processing.jog_march.straight_rules import straight_metrics
from shared_motion.training.catalog import COMMAND_RANGES
from shared_motion.training.catalog import TASK_NAMES
from shared_motion.training.catalog import measure
from shared_motion.training.data import MotionDataset
from shared_motion.training.data import assert_disjoint
from shared_motion.training.geometry import Skeleton
from shared_motion.training.model import file_sha256
from shared_motion.training.runner import save_json


@hydra.main(
    version_base="1.3", config_path="../../config", config_name="jog_march_repair"
)
def main(config):
    torch.set_num_threads(2)
    output = Path(config.output)
    records = json.loads((output / "running_index.json").read_text())
    provenance = json.loads((output / "provenance.json").read_text())
    for path, expected in provenance.items():
        assert file_sha256(path) == expected, path
    skeleton = Skeleton(config.skeleton)
    metadata = json.loads((output / "new_event_embeddings.json").read_text())
    new_embeddings = np.load(output / "new_event_embeddings.npz")
    embedding_root = (
        Path(config.project)
        / "datasets/annotations/frankenstein-dataset/text_embeddings/clip"
    )
    original_index = json.loads((embedding_root / "clip_index.json").read_text())
    original_slices = np.load(embedding_root / "clip_slice.npy")
    original_embeddings = np.load(embedding_root / "clip.npy", mmap_mode="r")
    pca = np.load(
        Path(config.project) / "outputs_amass/unified_direct_20261008/data/pca.npz"
    )
    parts = json.loads(
        (Path(config.project) / "pretrained/official/config.json").read_text()
    )["data"]["text_encoder"]["body_part_order"]
    intervals = defaultdict(list)
    maximum_error = 0.0
    for record in records:
        begin, end = record["crop_start_frame_20fps"], record["crop_end_frame_20fps"]
        original = np.load(record["motion_source"], mmap_mode="r")[begin:end].copy()
        with np.load(record["cache"]) as cache:
            assert np.array_equal(original, cache["motion"])
            assert file_sha256(record["cache"]) == record["cache_sha256"]
            assert (
                record["real_frames"] == len(original) == record["target_frames"]
                and record["pad_frames"] == 0
            )
            assert int(cache["task"]) == TASK_NAMES.index(record["task"])
            with torch.no_grad():
                joints = skeleton(torch.from_numpy(original)[None])[0].numpy()
                value = float(
                    measure(
                        skeleton,
                        torch.from_numpy(original)[None],
                        torch.tensor([record["task_id"]]),
                        torch.tensor([len(original)]),
                    )[0]
                )
            task, metrics, failures = classify_clip(joints, config.rules)
            assert not failures and task == record["task"], (record["key"], failures)
            if task == "jog" and config.get("straight"):
                straight, failures = straight_metrics(joints, config.straight)
                assert not failures, (record["key"], failures)
                assert record["semantic_revision"] == "straight_running_v2"
                for name, value_in_record in record["straight_metrics"].items():
                    assert abs(straight[name] - value_in_record) < 1e-5
            if task == "jog":
                # Rotations preserve XY speed; this avoids relying on FK/measure.
                independent = float(
                    np.linalg.norm(original[:-1, 1:3], axis=1).mean() * 20
                )
            else:
                feet = joints[:, [7, 8], 2]
                independent = float(
                    (feet.max(axis=0) - np.quantile(feet, 0.05, axis=0)).mean()
                )
            maximum_error = max(
                maximum_error,
                abs(value - independent),
                abs(value - float(cache["quantity"])),
            )
            assert abs(value - record["quantity"]) < 1e-6
            text = record["caption"]
            if text in original_index:
                lower, upper = original_slices[original_index[text]]
                expected = original_embeddings[lower:upper][0]
            else:
                expected = new_embeddings[str(metadata["labels"].index(text))]
            assert np.array_equal(cache["tx"], expected)
            reduced = (expected - pca["mean"]) @ pca["components"].T
            for part_index, part in enumerate(parts):
                block = slice(part_index * 51, (part_index + 1) * 51)
                active = part in ["action", "left_leg", "right_leg"]
                assert (cache["local_mask"][:, block] == active).all()
                assert np.allclose(
                    cache["local"][:, block],
                    reduced if active else np.zeros_like(reduced),
                    atol=1e-5,
                )
            assert all(
                np.isfinite(cache[name]).all()
                for name in ["motion", "quantity", "local", "tx"]
            )
        for prior_begin, prior_end, prior_task in intervals[record["family"]]:
            assert min(end, prior_end) <= max(begin, prior_begin), (
                record["family"],
                prior_task,
                task,
            )
        intervals[record["family"]].append((begin, end, task))
    assert maximum_error < 1e-5
    manifests = {}
    for split in ["train", "val"]:
        before = json.loads((Path(config.base) / f"{split}.json").read_text())
        after = json.loads((output / f"{split}.json").read_text())
        assert [
            record for record in before if record["task"] not in ["jog", "march"]
        ] == [record for record in after if record["task"] not in ["jog", "march"]]
        assert [record for record in after if record["task"] in ["jog", "march"]] == [
            record for record in records if record["split"] == split
        ]
        if config.get("straight"):
            assert [record for record in before if record["task"] != "jog"] == [
                record for record in after if record["task"] != "jog"
            ]
        manifests[split] = after
    assert not (
        {record["family"] for record in manifests["train"]}
        & {record["family"] for record in manifests["val"]}
    )
    training = MotionDataset(
        output / "train.json", "train", ["jog", "march"], config.project
    )
    validation = MotionDataset(
        output / "val.json", "val", ["jog", "march"], config.project
    )
    assert_disjoint(training, validation)
    report = dict(
        verified_caches=len(records),
        source_crops_exact=True,
        independent_quantity_max_error=maximum_error,
        all_admission_checks_recomputed=True,
        text_pca_masks_verified=True,
        other18_unchanged=True,
        march_unchanged=bool(config.get("straight")),
        straight_admission_verified=bool(config.get("straight")),
        all20_split_isolated=True,
        no_overlapping_cross_task_or_duplicate_crops=True,
        input_hashes_verified=len(provenance),
        training_started=False,
        tasks={},
    )
    for task in ["jog", "march"]:
        bounds = COMMAND_RANGES[TASK_NAMES.index(task)]
        task_rows = [record for record in records if record["task"] == task]
        summary = dict(
            bounds=bounds,
            source_kinds=dict(Counter(record["source_kind"] for record in task_rows)),
            annotation_levels=dict(
                Counter(record["annotation_level"] for record in task_rows)
            ),
            datasets=dict(
                Counter(record["family"].split("/")[0] for record in task_rows)
            ),
        )
        for split in ["train", "val"]:
            selected = [record for record in task_rows if record["split"] == split]
            values = [record["quantity"] for record in selected]
            hist, _ = np.histogram(values, bins=np.linspace(*bounds, 11))
            summary[split] = dict(
                samples=len(selected),
                families=len({record["family"] for record in selected}),
                range=[min(values), max(values)],
                inside_range=sum(bounds[0] <= value <= bounds[1] for value in values),
                bin_counts=hist.tolist(),
                empty_bins=np.flatnonzero(hist == 0).tolist(),
            )
        summary["maximum_root_excursion_m"] = max(
            record["metrics"]["root_excursion_m"] for record in task_rows
        )
        summary["mean_speed_range"] = [
            min(record["metrics"]["mean_path_speed_m_s"] for record in task_rows),
            max(record["metrics"]["mean_path_speed_m_s"] for record in task_rows),
        ]
        summary["from_old_jog_families"] = sum(
            "jog" in record["old_task_memberships"] for record in task_rows
        )
        report["tasks"][task] = summary
    save_json(output / "verification.json", report)
    repository = Path(__file__).resolve().parents[2]
    snapshot = {}
    for path in list(Path(__file__).parent.glob("*.py")) + [
        repository / "config/jog_march_repair.yaml",
        repository / "config/jog_straight_repair.yaml",
        repository / "shared_motion/training/catalog.py",
        repository / "shared_motion/training/geometry.py",
        repository / "data_processing/event_text.py",
        repository / "scripts/prepare_amass20.py",
        repository / "src/tools/geometry.py",
    ]:
        relative = path.relative_to(repository)
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        snapshot[str(relative)] = file_sha256(destination)
    save_json(output / "source_snapshot.json", snapshot)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
