"""Freeze source provenance and independently check mirrored inputs/text/splits."""

import json
from pathlib import Path
import shutil

import hydra
import numpy as np
from scipy.signal import find_peaks

from data_processing.sidestep.rules import mirror_body
from shared_motion.training.model import file_sha256
from shared_motion.training.runner import save_json


@hydra.main(
    version_base="1.3", config_path="../../config", config_name="sidestep_repair"
)
def main(config):
    output = Path(config.output)
    repository = Path(__file__).parents[2]
    project = Path(config.project)
    records = json.loads((output / "sidestep_index.json").read_text())
    events = json.loads((output / "recovered_events.json").read_text())
    sources = json.loads((output / "sources.json").read_text())
    # The parameterized closure detector must reproduce the executed fixed-value
    # detector exactly before its refactoring is accepted as the source snapshot.
    for event in events:
        family = event["family"]
        separation = np.load(
            output / "inspection" / f'{Path(sources[family]["raw"]).stem}.npz'
        )["ankle_separation"]
        start = event["event"]["start"]
        stop = event["event"]["stop"]
        closures = (
            find_peaks(
                -separation[start:stop],
                prominence=config.rules.closure_prominence_m,
                distance=config.rules.closure_distance_frames,
            )[0]
            + start
        )
        assert (
            sorted(set([start, *closures.tolist(), stop - 1]))
            == event["closure_boundaries_20fps"]
        )
    for source in sources.values():
        assert file_sha256(source["raw"]) == source["raw_sha256"]
        assert file_sha256(source["motion"]) == source["motion_sha256"]
        assert file_sha256(source["mirrored_raw"]) == source["mirrored_raw_sha256"]
        assert (
            file_sha256(source["mirrored_motion"]) == source["mirrored_motion_sha256"]
        )
        with np.load(source["raw"]) as original, np.load(
            source["mirrored_raw"]
        ) as mirrored:
            expected_poses, expected_translation = mirror_body(
                original["poses"], original["trans"]
            )
            np.testing.assert_array_equal(expected_poses, mirrored["poses"])
            np.testing.assert_array_equal(expected_translation, mirrored["trans"])
    original_annotations = (
        project / "datasets/annotations/frankenstein-dataset/annotations"
    )
    all_manifests = {
        split: json.loads((output / f"{split}.json").read_text())
        for split in ["train", "val"]
    }
    families = {
        split: {record["family"] for record in selected}
        for split, selected in all_manifests.items()
    }
    assert not families["train"] & families["val"]
    native_index = json.loads((output / "native_left_index.json").read_text())
    assert all(
        record["task"] == "sidestep_native_left_reference"
        and not record["ready_for_training"]
        for record in native_index
    )
    pca_path = project / "outputs_amass/unified_direct_20261008/data/pca.npz"
    pca = np.load(pca_path)
    official_config = project / "pretrained/official/config.json"
    parts = json.loads(official_config.read_text())["data"]["text_encoder"][
        "body_part_order"
    ]
    new_labels = json.loads((output / "new_event_embeddings.json").read_text())[
        "labels"
    ]
    embeddings = np.load(output / "new_event_embeddings.npz")
    for record in records:
        for cache_path, label, task_index in [
            (record["cache"], record["caption"], 5),
            (
                record["native_left_cache"],
                "side step to the left without crossing the feet",
                -1,
            ),
        ]:
            embedding = embeddings[str(new_labels.index(label))]
            reduced = ((embedding - pca["mean"]) @ pca["components"].T).astype(
                np.float32
            )
            with np.load(cache_path) as cache:
                assert int(cache["task"]) == task_index
                np.testing.assert_array_equal(cache["tx"], embedding)
                for part_index, part in enumerate(parts):
                    active = part in ["action", "left_leg", "right_leg"]
                    assert np.all(
                        cache["local_mask"][:, part_index * 51 : (part_index + 1) * 51]
                        == active
                    )
                    expected = np.broadcast_to(
                        reduced if active else np.zeros_like(reduced),
                        (len(cache["motion"]), 51),
                    )
                    np.testing.assert_array_equal(
                        cache["local"][:, part_index * 51 : (part_index + 1) * 51],
                        expected,
                    )
    inputs = [
        config.skeleton,
        pca_path,
        official_config,
        project
        / "outputs_amass/stage2_manifests_20261008/annotated_source_raw_audit.json",
        original_annotations / "annotations.json",
    ]
    inputs.extend(
        original_annotations / "splits" / f"{split}.txt"
        for split in ["train", "val", "test"]
    )
    inputs.extend(
        Path(config.babel) / name
        for name in ["train.json", "val.json", "extra_train.json", "extra_val.json"]
    )
    inputs.extend(Path(config.base) / f"{split}.json" for split in ["train", "val"])
    inputs.append("/mnt/sda2/dataset/mdm_refine_phc_mvp_v1/assets/clip/ViT-B-32.pt")
    files = list((repository / "data_processing/sidestep").glob("*.py"))
    files.extend(
        repository / name
        for name in [
            "data_processing/event_text.py",
            "data_processing/export_task_audits.py",
            "data_processing/parameter_coverage.py",
            "scripts/prepare_amass20.py",
            "shared_motion/training/geometry.py",
            "shared_motion/training/catalog.py",
            "shared_motion/adapter/kinematics.py",
            "src/tools/smplrifke_feats.py",
            "src/tools/geometry.py",
            "config/sidestep_repair.yaml",
        ]
    )
    inputs.extend(files)
    external = [
        project / "work/unified_direct/prepare_data.py",
        project / "work/unified_direct/common.py",
        project / "prepare/amasstools/fix_fps.py",
    ]
    inputs.extend(external)
    snapshot = output / "source"
    snapshot.mkdir(exist_ok=True)
    for source_path in files:
        destination = snapshot / source_path.relative_to(repository)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, destination)
    for source_path in external:
        destination = snapshot / "external" / source_path.relative_to(project)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source_path, destination)
    save_json(
        output / "provenance.json", {str(path): file_sha256(path) for path in inputs}
    )
    save_json(
        output / "final_verification.json",
        dict(
            mirror_raw_exact=True,
            closure_config_refactor_matches_executed_boundaries=True,
            text_embeddings_pca_and_masks_exact=True,
            all20_source_families_split_disjoint=True,
            native_left_cannot_match_current_task_loader=True,
            raw_and_motion_hashes_verified=True,
            cache_pairs=len(records),
            rejected_non_test_cycles=1,
            held_out_test_families=1,
        ),
    )
    print("Final input, mirror, text, split and provenance checks passed", flush=True)


if __name__ == "__main__":
    main()
