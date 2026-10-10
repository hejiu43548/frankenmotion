"""Rebuild the selected real strike crops and label metric wrist endpoints.

Selection/crop boundaries come from the colleague's 13/1 cleaned-data manifests.
Motion arrays and CLIP features are rebuilt on the current host, not claimed to
be byte-identical to unavailable original NPZ caches. No teacher is involved.
"""

import json
from pathlib import Path
import sys

import clip
import hydra
import numpy as np
from omegaconf import OmegaConf
from sklearn.decomposition import PCA
import sklearn
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared_motion.training.geometry import Skeleton
from shared_motion.training.catalog import measure
from shared_motion.training.model import file_sha256
from shared_motion.training.reach3d_geometry import wrist_positions_in_body_frame


@hydra.main(version_base="1.3", config_path="../config", config_name="prepare_strike3d")
def main(config):
    torch.set_num_threads(4)
    np.random.seed(config.pca_seed)
    root = Path(config.root)
    output = Path(config.output)
    (output / "cache").mkdir(parents=True, exist_ok=True)
    manifests = {
        split: json.loads(
            (Path(config.source_manifest_directory) / f"{split}.json").read_text()
        )
        for split in ["train", "val"]
    }
    rows = manifests["train"] + manifests["val"]
    assert len(manifests["train"]) == 13 and len(manifests["val"]) == 1
    assert not (
        {row["family"] for row in manifests["train"]}
        & {row["family"] for row in manifests["val"]}
    )
    embeddings_root = (
        root / "datasets/annotations/frankenstein-dataset/text_embeddings/clip"
    )
    embeddings = np.load(embeddings_root / "clip.npy")
    slices = np.load(embeddings_root / "clip_slice.npy")
    index = json.loads((embeddings_root / "clip_index.json").read_text())
    projection = PCA(n_components=51).fit(embeddings)
    np.savez(
        output / "pca.npz", mean=projection.mean_, components=projection.components_
    )
    labels = sorted({row["caption"] for row in rows})
    missing = [label for label in labels if label not in index]
    extra = {}
    parity = {}
    if missing:
        encoder, _ = clip.load(config.clip_checkpoint, device="cpu")
        encoder.eval()
        checks = ["punch", "right punch"]
        with torch.no_grad():
            encoded = (
                encoder.encode_text(clip.tokenize(missing + checks)).float().numpy()
            )
        extra = dict(zip(missing, encoded[: len(missing)]))
        for label, value in zip(checks, encoded[len(missing) :]):
            original = embeddings[slices[index[label]][0]]
            cosine = float(
                np.dot(original, value)
                / (np.linalg.norm(original) * np.linalg.norm(value))
            )
            if cosine < 0.9999:
                raise ValueError(f"CLIP parity failed: {label}: {cosine}")
            parity[label] = cosine
    skeleton = Skeleton(config.skeleton)
    parts = json.loads((root / "pretrained/official_20260910/config.json").read_text())[
        "data"
    ]["text_encoder"]["body_part_order"]
    prepared = []
    for row in rows:
        source = (
            root
            / "datasets/motions/AMASS_20.0_fps_nh_smplrifke"
            / (row["family"] + ".npy")
        )
        start, stop = row["crop_start_frame_20fps"], row["crop_end_frame_20fps"]
        motion = np.asarray(np.load(source, mmap_mode="r")[start:stop]).copy()
        assert motion.shape == (stop - start, 205) and np.isfinite(motion).all()
        tensor = torch.from_numpy(motion).float()[None]
        with torch.no_grad():
            joints = skeleton(tensor)
            wrists = wrist_positions_in_body_frame(joints[..., :22, :])[0]
            quantity = float(
                measure(
                    skeleton, tensor, torch.tensor([2]), torch.tensor([len(motion)])
                )[0]
            )
        if abs(quantity - row["quantity"]) > 1e-4:
            raise ValueError(f'Historical crop metric mismatch: {row["key"]}')
        peak = row["local_peak_frame"]
        endpoint_stop = min(
            len(motion),
            peak + int(config.endpoint_after_peak_frames) + 1,
            int(np.floor(row["source_label"]["end_t"] * 20)) - start + 1,
        )
        if endpoint_stop <= peak:
            raise ValueError("No valid post-peak labeled endpoint interval")
        event_frame = peak + int(wrists[peak:endpoint_stop, 1, 0].argmax())
        target = wrists[event_frame, 1].numpy()
        label = row["caption"]
        text = (
            extra[label]
            if label in extra
            else np.asarray(embeddings[slices[index[label]][0]])
        )
        reduced = (text - projection.mean_) @ projection.components_.T
        local = np.zeros((len(motion), 408), np.float32)
        local_mask = np.zeros_like(local, dtype=bool)
        times = np.arange(start, stop) / 20
        active = (times >= row["source_label"]["start_t"]) & (
            times < row["source_label"]["end_t"]
        )
        for part_index, part in enumerate(parts):
            if part in ["action", "right_arm"]:
                local[active, part_index * 51 : (part_index + 1) * 51] = reduced
                local_mask[active, part_index * 51 : (part_index + 1) * 51] = True
        cache = output / "cache" / (row["key"] + ".npz")
        np.savez(
            cache,
            motion=motion,
            local=local,
            local_mask=local_mask,
            tx=text,
            hands=np.int64(1),
            positions=target,
            event_frames=np.int64(event_frame),
        )
        prepared.append(
            dict(
                row,
                cache=str(cache),
                rebuilt_cache_sha256=file_sha256(cache),
                original_cache_sha256=row["cache_sha256"],
                source_motion_sha256=file_sha256(source),
                event_frame=event_frame,
                target_xyz_m=target.tolist(),
                endpoint_interval=[peak, endpoint_stop],
                endpoint_at_search_boundary=event_frame in [peak, endpoint_stop - 1],
            )
        )
    for split in manifests:
        (output / f"{split}.json").write_text(
            json.dumps([row for row in prepared if row["split"] == split], indent=2)
            + "\n"
        )
    targets = np.array(
        [row["target_xyz_m"] for row in prepared if row["split"] == "train"]
    )
    report = dict(
        counts={split: len(rows) for split, rows in manifests.items()},
        train_target_min=targets.min(0).tolist(),
        train_target_max=targets.max(0).tolist(),
        train_target_center=targets.mean(0).tolist(),
        train_target_scale=np.maximum(targets.std(0), 0.05).tolist(),
        clip_parity=parity,
        sklearn_version=sklearn.__version__,
        pca_solver=projection._fit_svd_solver,
        pca_sha256=file_sha256(output / "pca.npz"),
        skeleton_sha256=file_sha256(config.skeleton),
        endpoint_definition="Maximum body-forward wrist coordinate in the labeled interval from outbound speed peak to peak+8 frames. Automatic geometric endpoint, not verified contact ground truth.",
        boundary_events=[
            row["key"] for row in prepared if row["endpoint_at_search_boundary"]
        ],
        source_manifest_hashes={
            split: file_sha256(Path(config.source_manifest_directory) / f"{split}.json")
            for split in manifests
        },
    )
    (output / "audit.json").write_text(json.dumps(report, indent=2) + "\n")
    OmegaConf.save(config, output / "config.yaml")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
