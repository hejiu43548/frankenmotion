"""Verify every final jump crop, conditioning array, split and quantity."""

from collections import Counter
import json
from pathlib import Path
import shutil

import hydra
import numpy as np
import torch

from data_processing.jump.rules import check_clip
from shared_motion.training.catalog import HUMAN_HEIGHT
from shared_motion.training.data import MotionDataset
from shared_motion.training.data import assert_disjoint
from shared_motion.training.geometry import Skeleton
from shared_motion.training.model import file_sha256
from shared_motion.training.runner import save_json


@hydra.main(version_base="1.3", config_path="../../config", config_name="jump_repair")
def main(config):
    torch.set_num_threads(2)
    output = Path(config.output)
    records = json.loads((output / "jump_index.json").read_text())
    provenance = json.loads((output / "provenance.json").read_text())
    for path, expected in provenance.items():
        assert file_sha256(path) == expected, path
    skeleton = Skeleton(config.skeleton)
    metadata = json.loads((output / "new_event_embeddings.json").read_text())
    embeddings = np.load(output / "new_event_embeddings.npz")
    pca = np.load(
        Path(config.project) / "outputs_amass/unified_direct_20261008/data/pca.npz"
    )
    clip_root = (
        Path(config.project)
        / "datasets/annotations/frankenstein-dataset/text_embeddings/clip"
    )
    original_index = json.loads((clip_root / "clip_index.json").read_text())
    original_slices = np.load(clip_root / "clip_slice.npy")
    original_text = np.load(clip_root / "clip.npy", mmap_mode="r")
    official_path = Path(config.project) / "pretrained/official/config.json"
    parts = json.loads(official_path.read_text())["data"]["text_encoder"][
        "body_part_order"
    ]
    maximum_error = 0.0
    for record in records:
        begin, end = record["crop_start_frame_20fps"], record["crop_end_frame_20fps"]
        original = np.load(record["motion_source"], mmap_mode="r")[begin:end]
        with np.load(record["cache"]) as archive:
            assert np.array_equal(original, archive["motion"])
            assert file_sha256(record["cache"]) == record["cache_sha256"]
            assert len(original) == record["real_frames"] and record["pad_frames"] == 0
            # Root height is feature0; this recomputation is independent of FK/measure.
            quantity = float(
                (original[:, 0].max() - original[0, 0])
                * (HUMAN_HEIGHT / skeleton.height)
            )
            maximum_error = max(
                maximum_error, abs(quantity - float(archive["quantity"]))
            )
            assert abs(quantity - record["quantity"]) < 1e-6
            assert int(archive["task"]) == 8
            label = record["caption"]
            if label in original_index:
                start_text, stop_text = original_slices[original_index[label]]
                expected_text = original_text[start_text:stop_text][0]
            else:
                expected_text = embeddings[str(metadata["labels"].index(label))]
            assert np.array_equal(archive["tx"], expected_text)
            reduced = (expected_text - pca["mean"]) @ pca["components"].T
            for part_index, part in enumerate(parts):
                active = part in ["action", "left_leg", "right_leg"]
                block = slice(part_index * 51, (part_index + 1) * 51)
                assert (archive["local_mask"][:, block] == active).all()
                assert np.allclose(
                    archive["local"][:, block],
                    reduced if active else np.zeros_like(reduced),
                    atol=1e-5,
                )
            with torch.no_grad():
                joints = skeleton(torch.from_numpy(original.copy())[None])[
                    0
                ].numpy() * (HUMAN_HEIGHT / skeleton.height)
            assert not check_clip(joints, config.rules)[1], record["key"]
    assert maximum_error < 1e-6
    manifests = {}
    old_jump = {}
    coverage = {}
    for split in ["train", "val"]:
        before = json.loads((Path(config.base) / f"{split}.json").read_text())
        after = json.loads((output / f"{split}.json").read_text())
        assert [record for record in before if record["task"] != "jump"] == [
            record for record in after if record["task"] != "jump"
        ]
        assert [record for record in after if record["task"] == "jump"] == [
            record for record in records if record["split"] == split
        ]
        manifests[split] = after
        old_jump[split] = [record for record in before if record["task"] == "jump"]
        selected = [record for record in records if record["split"] == split]
        values = [record["quantity"] for record in selected]
        hist, _ = np.histogram(values, bins=np.linspace(0.25, 0.55, 11))
        coverage[split] = dict(
            samples=len(values),
            minimum=min(values),
            maximum=max(values),
            inside_range=sum(0.25 <= value <= 0.55 for value in values),
            bin_counts=hist.tolist(),
            empty_bins=np.flatnonzero(hist == 0).tolist(),
        )
    assert not (
        {record["family"] for record in manifests["train"]}
        & {record["family"] for record in manifests["val"]}
    )
    training = MotionDataset(output / "train.json", "train", ["jump"], config.project)
    validation = MotionDataset(output / "val.json", "val", ["jump"], config.project)
    assert_disjoint(training, validation)
    old_families = {record["family"] for rows in old_jump.values() for record in rows}
    new_families = {record["family"] for record in records}
    verification = dict(
        definition="In-place two-foot jump with takeoff and landing at the same location",
        maximum_landing_center_offset_m=max(
            record["metrics"]["landing_center_offset_m"] for record in records
        ),
        maximum_each_foot_landing_offset_m=max(
            record["metrics"]["each_foot_landing_offset_m"] for record in records
        ),
        maximum_root_horizontal_excursion_m=max(
            record["metrics"]["root_horizontal_excursion_m"] for record in records
        ),
        verified_caches=len(records),
        source_crops_exact=True,
        all_input_hashes_verified=len(provenance),
        independent_quantity_max_error=maximum_error,
        text_pca_masks_verified=True,
        all_admission_checks_recomputed=True,
        other19_tasks_unchanged=True,
        all20_family_split_isolated=True,
        old_counts={split: len(rows) for split, rows in old_jump.items()},
        counts=dict(train=len(training.rows), val=len(validation.rows)),
        families={
            split: len(
                {record["family"] for record in records if record["split"] == split}
            )
            for split in ["train", "val"]
        },
        datasets=dict(Counter(record["family"].split("/")[0] for record in records)),
        reused_old_jump_families=len(old_families & new_families),
        added_category_backed_families=len(new_families - old_families),
        removed_old_jump_families=len(old_families - new_families),
        command_range=[0.25, 0.55],
        coverage=coverage,
        training_started=False,
        limitations=[
            "Kinematic support is an estimated proxy, not measured ground-contact truth.",
            "Single-action sequence labels have no official event boundaries; all crops are motion-derived.",
            "HDM05 native cuts could not be verified; no category coverage claim is used as admission.",
            "Automatic admission and ten-sample visual review do not establish exhaustive human semantic acceptance.",
        ],
    )
    save_json(output / "verification.json", verification)
    repository = Path(__file__).resolve().parents[2]
    sources = list((repository / "data_processing/jump").glob("*.py")) + [
        repository / "data_processing/event_text.py",
        repository / "config/jump_repair.yaml",
        repository / "scripts/prepare_amass20.py",
        repository / "shared_motion/training/geometry.py",
        repository / "shared_motion/training/catalog.py",
        repository / "shared_motion/adapter/kinematics.py",
        repository / "src/tools/geometry.py",
    ]
    source_hashes = {}
    for source in sources:
        relative = source.relative_to(repository)
        destination = output / "source" / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, destination)
        source_hashes[str(relative)] = file_sha256(destination)
    save_json(output / "source_snapshot.json", source_hashes)
    print(json.dumps(verification, indent=2), flush=True)


if __name__ == "__main__":
    main()
