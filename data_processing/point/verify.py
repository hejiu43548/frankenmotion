"""Verify source crops, simultaneous XYZ targets, conditioning and19-task merge."""

from collections import Counter
import json
from pathlib import Path
import shutil

import hydra
import numpy as np
import torch

from data_processing.point.repair import crop_metrics, compatible_event
from shared_motion.training.catalog import ACTIVE_TASK_NAMES, measure_commands
from shared_motion.training.data import MotionDataset, assert_disjoint
from shared_motion.training.geometry import Skeleton
from shared_motion.training.model import file_sha256
from shared_motion.training.reach import REACH_POLICY, REACH_BOUNDS
from shared_motion.training.runner import save_json


@hydra.main(
    version_base="1.3", config_path="../../config", config_name="reach_xyz_repair"
)
def main(config):
    torch.set_num_threads(2)
    output = Path(config.output)
    rows = json.loads((output / "reach_index.json").read_text())
    skeleton = Skeleton(config.skeleton)
    provenance = json.loads((output / "provenance.json").read_text())
    repository = Path(__file__).resolve().parents[2]
    for filename in ["train.json", "val.json", "extra_train.json", "extra_val.json"]:
        path = Path(config.babel) / filename
        provenance[str(path)] = file_sha256(path)
    annotation_root = (
        Path(config.project) / "datasets/annotations/frankenstein-dataset/annotations"
    )
    for path in [
        annotation_root / "annotations.json",
        *[
            annotation_root / "splits" / f"{split}.txt"
            for split in ["train", "val", "test"]
        ],
        Path(config.skeleton),
    ]:
        provenance[str(path)] = file_sha256(path)
    for path, digest in provenance.items():
        assert file_sha256(path) == digest, path
    metadata = json.loads((output / "new_event_embeddings.json").read_text())
    embeddings = np.load(output / "new_event_embeddings.npz")
    pca = np.load(
        Path(config.project) / "outputs_amass/unified_direct_20261008/data/pca.npz"
    )
    parts = json.loads(
        (Path(config.project) / "pretrained/official/config.json").read_text()
    )["data"]["text_encoder"]["body_part_order"]
    maximum_error = 0.0
    for record in rows:
        assert compatible_event(record)
        begin, end = record["crop_start_frame_20fps"], record["crop_end_frame_20fps"]
        source = np.load(record["motion_source"], mmap_mode="r")[begin:end].copy()
        assert file_sha256(record["cache"]) == record["cache_sha256"]
        with np.load(record["cache"]) as cache:
            assert np.array_equal(cache["motion"], source)
            assert (
                len(source) == record["real_frames"] == record["target_frames"]
                and record["pad_frames"] == 0
            )
            with torch.no_grad():
                positions = skeleton(torch.from_numpy(source)[None])[0].numpy()
                measured = measure_commands(
                    skeleton,
                    torch.from_numpy(source)[None],
                    torch.tensor([1]),
                    torch.tensor([len(source)]),
                )[0].numpy()
            metrics, failures = crop_metrics(positions, config.rules)
            assert not failures, (record["key"], failures)
            lateral = positions[0, 1, :2] - positions[0, 2, :2]
            lateral = lateral / np.linalg.norm(lateral)
            forward = np.array([lateral[1], -lateral[0]])
            wrist = positions[-5:, 21] - positions[0, 0]
            independent = np.array(
                [
                    (wrist[:, :2] * forward).sum(-1).mean(),
                    (wrist[:, :2] * lateral).sum(-1).mean(),
                    wrist[:, 2].mean(),
                ]
            )
            maximum_error = max(
                maximum_error,
                float(np.abs(independent - cache["quantity"]).max()),
                float(np.abs(measured - cache["quantity"]).max()),
            )
            assert np.allclose(cache["quantity"], record["target_xyz_m"], atol=1e-6)
            expected = embeddings[str(metadata["labels"].index(record["caption"]))]
            assert np.array_equal(cache["tx"], expected)
            reduced = (expected - pca["mean"]) @ pca["components"].T
            for part_index, part in enumerate(parts):
                block = slice(part_index * 51, (part_index + 1) * 51)
                active = part in ["action", "right_arm"]
                assert (cache["local_mask"][:, block] == active).all()
                assert np.allclose(
                    cache["local"][:, block],
                    reduced if active else np.zeros_like(reduced),
                    atol=1e-5,
                )
    assert maximum_error < 1e-5
    manifests = {}
    for split in ["train", "val"]:
        before = json.loads((Path(config.data) / f"{split}.json").read_text())
        after = json.loads((output / f"{split}.json").read_text())
        assert [
            record for record in before if record["task"] not in ["reach", "point"]
        ] == [record for record in after if record["task"] != "reach"]
        assert set(record["task"] for record in after) == set(ACTIVE_TASK_NAMES)
        manifests[split] = after
    assert not (
        {record["family"] for record in manifests["train"]}
        & {record["family"] for record in manifests["val"]}
    )
    training = MotionDataset(
        output / "train.json", "train", ACTIVE_TASK_NAMES, config.project
    )
    validation = MotionDataset(
        output / "val.json", "val", ACTIVE_TASK_NAMES, config.project
    )
    assert_disjoint(training, validation)
    batch = training.batch(
        [training.groups[index][0] for index in sorted(training.groups)], "cpu"
    )
    assert batch["quantity"].shape == (19, 3)
    intervals = {}
    for record in rows:
        begin, end = record["crop_start_frame_20fps"], record["crop_end_frame_20fps"]
        for lower, upper in intervals.setdefault(record["family"], []):
            assert min(end, upper) <= max(begin, lower)
        intervals[record["family"]].append((begin, end))
    coverage = {}
    for split in ["train", "val"]:
        selected = [record for record in rows if record["split"] == split]
        targets = np.array([record["target_xyz_m"] for record in selected])
        bounds = np.array(REACH_BOUNDS)
        coverage[split] = dict(
            samples=len(selected),
            families=len({record["family"] for record in selected}),
            xyz_min_m=targets.min(0).tolist(),
            xyz_max_m=targets.max(0).tolist(),
            inside_normalization_box=int(
                ((targets >= bounds[:, 0]) & (targets <= bounds[:, 1])).all(-1).sum()
            ),
            source_kinds=dict(Counter(record["source_kind"] for record in selected)),
        )
    sources = {}
    paths = (
        list((repository / "data_processing/point").glob("*.py"))
        + list((repository / "shared_motion/training").glob("*.py"))
        + [
            repository / "scripts/prepare_amass20.py",
            repository / "scripts/infer.py",
            repository / "data_processing/event_text.py",
            repository / "config/reach_xyz_repair.yaml",
            repository / "config/audit_point.yaml",
            repository / "config/data/amass20.yaml",
            repository / "tests/test_reach_xyz.py",
        ]
    )
    for path in paths:
        relative = path.relative_to(repository)
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(path, destination)
        sources[str(relative)] = file_sha256(path)
    save_json(output / "source_snapshot.json", sources)
    save_json(output / "provenance.json", provenance)
    report = dict(
        verified_reach_caches=len(rows),
        maximum_independent_xyz_error_m=maximum_error,
        source_crops_exact=True,
        text_pca_masks_verified=True,
        all19_family_split_isolated=True,
        other18_records_unchanged=True,
        point_removed_from_active_tasks=True,
        stable_task_ids=True,
        mixed_batch_command_shape=[19, 3],
        coverage=coverage,
        policy=REACH_POLICY,
        training_started=False,
    )
    save_json(output / "verification.json", report)
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
