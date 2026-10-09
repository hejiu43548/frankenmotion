#!/usr/bin/env python3
"""Wave repair v2: native HDM05 supplemented events and BABEL act_cat frames."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
import numpy as np
import torch
from scipy.signal import find_peaks

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.prepare_amass20 import convert_source
from shared_motion.training.geometry import Skeleton
from shared_motion.training.catalog import measure
from shared_motion.training.model import file_sha256 as sha
from shared_motion.training.runner import save_json
from data_processing.wave.rules import metrics, motion_reasons, POLICY

ALLOWED_CATEGORIES = {
    "wave",
    "gesture",
    "greet",
    "hand movements",
    "arm movements",
    "raising body part",
    "stand",
}
HARMLESS_OVERLAP = ALLOWED_CATEGORIES | {
    "look",
    "head movements",
    "turn head",
    "move body part",
}
ADMISSION = dict(
    POLICY, root_excursion_m=0.30, root_mean_speed_m_s=0.35, foot_excursion_m=0.15
)


def waveform(j):
    side = j[:, 16] - j[:, 17]
    side /= np.maximum(np.linalg.norm(side, axis=-1, keepdims=True), 1e-8)
    lateral = ((j[:, 21] - (j[:, 16] + j[:, 17]) / 2) * side).sum(-1)
    smooth = np.convolve(
        np.pad(lateral, (2, 2), mode="edge"), np.ones(5) / 5, mode="valid"
    )
    peaks = len(find_peaks(smooth, prominence=0.04, distance=4)[0]) + len(
        find_peaks(-smooth, prominence=0.04, distance=4)[0]
    )
    return dict(
        lateral_half_range_m=float(
            (np.quantile(lateral, 0.95) - np.quantile(lateral, 0.05)) / 2
        ),
        lateral_reversals=peaks,
        right_wrist_raised_fraction=float(((j[:, 21, 2] - j[:, 0, 2]) > 0.2).mean()),
    )


def canonical(path):
    # BABEL feat_p contains a dataset wrapper before the AMASS-relative path.
    return str(Path(*Path(path).parts[1:]).with_suffix(""))


def chunks(start, stop):
    if stop - start < 40:
        return []
    edges = np.linspace(start, stop, int(np.ceil((stop - start) / 120)) + 1, dtype=int)
    return [(int(s), int(e)) for s, e in zip(edges[:-1], edges[1:])]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--project", type=Path, default=Path("/home/psirobot/projects/frankenmotion")
    )
    ap.add_argument("--babel", type=Path, default=Path("/mnt/sda2/dataset/babel"))
    ap.add_argument(
        "--base",
        type=Path,
        default=Path(
            "/mnt/sda2/frankenmotion/outputs_amass/shared20_walking_data_20261009"
        ),
    )
    ap.add_argument(
        "--clip-checkpoint",
        type=Path,
        default=Path("/mnt/sda2/dataset/mdm_refine_phc_mvp_v1/assets/clip/ViT-B-32.pt"),
    )
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "cache").mkdir()
    root = args.project
    torch.set_num_threads(2)
    annroot = root / "datasets/annotations/frankenstein-dataset/annotations"
    anns = json.loads((annroot / "annotations.json").read_text())
    splitmap = {}
    frank = {}
    for split in ["train", "val", "test"]:
        for key in (annroot / "splits" / f"{split}.txt").read_text().split():
            family = anns[key]["path"]
            assert splitmap.setdefault(family, split) == split
            frank.setdefault(family, []).append(key)
    skpath = (
        root
        / "outputs_amass/transfer_charlie_20261008/snapshot/outputs_amass/franken_eleven_20261003/skeleton.npz"
    )
    sk = Skeleton(skpath)
    raw = json.loads(
        (
            root
            / "outputs_amass/stage2_manifests_20261008/annotated_source_raw_audit.json"
        ).read_text()
    )
    nativepath = Path(__file__).with_name("hdm05_events.json")
    native = json.loads(nativepath.read_text())
    exclude = {x["family"] for x in native["excluded_sources"]}
    candidates = []
    rejected = []
    sources = {}
    babel_records = {}
    provenance = {}
    for e in native["events"]:
        family = f"MPI_HDM05/{e['actor']}/HDM_{e['actor']}_05-01_{e['take']}_120_poses"
        candidates.append(
            dict(
                family=family,
                split=splitmap.get(family),
                source_kind="hdm05_script_supplement",
                start_s=e["start"],
                end_s=e["end"],
                label=e["label"],
                source_label=e,
                source_reference=native["reference"],
            )
        )
    for file in ["train.json", "val.json", "extra_train.json", "extra_val.json"]:
        d = json.loads((args.babel / file).read_text())
        provenance[str(args.babel / file)] = sha(args.babel / file)
        for key, r in d.items():
            family = canonical(r["feat_p"])
            blocks = (
                (r.get("frame_anns") or [])
                if file.startswith("extra")
                else [r.get("frame_ann")]
            )
            for block in blocks:
                if not block:
                    continue
                for label in block["labels"]:
                    if "wave" not in (label.get("act_cat") or []):
                        continue
                    rec = dict(
                        family=family,
                        split=splitmap.get(family),
                        source_kind="babel_act_cat_frame",
                        babel_file=file,
                        babel_sid=key,
                        babel_lid=block.get("babel_lid"),
                        seg_id=label["seg_id"],
                        start_s=label["start_t"],
                        end_s=label["end_t"],
                        label=label["proc_label"],
                        source_label=label,
                        babel_duration_s=r["dur"],
                    )
                    babel_records[(file, key, block.get("babel_lid"))] = r
                    reasons = []
                    if set(label["act_cat"]) - ALLOWED_CATEGORIES:
                        reasons.append("conflicting_act_cat")
                    # Left-only descriptions; left-to-right directions are not anatomical handedness.
                    if re.search(
                        r"\bleft (?:hand|arm)\b|^wave left$", rec["label"]
                    ) and not re.search(r"\bright (?:hand|arm)\b|both", rec["label"]):
                        reasons.append("left_only_label")
                    if re.search(
                        r"circular|circle|towards self|in and out", rec["label"]
                    ):
                        reasons.append("different_wave_geometry")
                    # Subtract concurrent conflicting intervals instead of trusting sequence categories.
                    mask = np.ones(max(0, int(np.ceil(r["dur"] * 20))), bool)
                    for other in block["labels"]:
                        cats = set(other.get("act_cat") or [])
                        if other["seg_id"] == label["seg_id"]:
                            continue
                        if not cats or cats - HARMLESS_OVERLAP:
                            s = max(0, int(np.floor(other["start_t"] * 20)))
                            e = min(len(mask), int(np.ceil(other["end_t"] * 20)))
                            mask[s:e] = False
                    s = max(0, int(np.ceil(label["start_t"] * 20)))
                    e = min(len(mask), int(np.floor(label["end_t"] * 20)))
                    usable = mask[s:e]
                    edges = np.diff(np.r_[False, usable, False].astype(int))
                    intervals = [
                        (s + a, s + b)
                        for a, b in zip(
                            np.flatnonzero(edges == 1), np.flatnonzero(edges == -1)
                        )
                        if b - a >= 40
                    ]
                    if not intervals:
                        reasons.append("no_2s_unconflicted_frame_interval")
                    if reasons:
                        rejected.append(dict(rec, reasons=reasons))
                        continue
                    for start, stop in intervals:
                        candidates.append(
                            dict(rec, start_s=start / 20, end_s=stop / 20)
                        )
    # Training-source boundaries supplement native scenes; native candidates go first in deduplication.
    er = root / "datasets/annotations/frankenstein-dataset/text_embeddings/clip"
    emb = np.load(er / "clip.npy", mmap_mode="r")
    slices = np.load(er / "clip_slice.npy")
    idx = json.loads((er / "clip_index.json").read_text())
    pcapath = root / "outputs_amass/unified_direct_20261008/data/pca.npz"
    pca = np.load(pcapath)
    official = root / "pretrained/official/config.json"
    parts = json.loads(official.read_text())["data"]["text_encoder"]["body_part_order"]
    reduced = {}
    new_embeddings = {}
    embedding_check = {}
    missing = sorted(
        {
            r["label"]
            for r in candidates
            if r["split"] in ["train", "val"] and r["label"] not in idx
        }
    )
    if missing:
        import clip as clip_library

        device = "cuda" if torch.cuda.is_available() else "cpu"
        assert args.clip_checkpoint.is_file(), args.clip_checkpoint
        provenance[str(args.clip_checkpoint)] = sha(args.clip_checkpoint)
        model, _ = clip_library.load(str(args.clip_checkpoint), device=device)
        model.eval()
        checks = ["wave", "wave right hand", "wave both arms"]
        with torch.no_grad():
            for offset in range(0, len(missing) + len(checks), 32):
                labels = (missing + checks)[offset : offset + 32]
                encoded = (
                    model.encode_text(clip_library.tokenize(labels).to(device))
                    .float()
                    .cpu()
                    .numpy()
                )
                for label, value in zip(labels, encoded):
                    if label in missing:
                        new_embeddings[label] = value
                    else:
                        begin, end = slices[idx[label]]
                        old = np.asarray(emb[begin:end])[0]
                        cosine = float(
                            np.dot(old, value)
                            / (np.linalg.norm(old) * np.linalg.norm(value))
                        )
                        assert cosine > 0.9999, (label, cosine)
                        embedding_check[label] = dict(
                            cosine=cosine,
                            max_absolute_error=float(np.abs(old - value).max()),
                        )
        np.savez(
            out / "new_event_embeddings.npz",
            **{str(i): new_embeddings[label] for i, label in enumerate(missing)},
        )
        save_json(
            out / "new_event_embeddings.json",
            dict(
                labels=missing,
                encoder="OpenAI CLIP ViT-B/32",
                device=device,
                cached_embedding_parity=embedding_check,
                labels_are_original_source_text=True,
            ),
        )
        del model

    def embed(label):
        if label in new_embeddings:
            value = new_embeddings[label][None]
        else:
            start, stop = slices[idx[label]]
            value = np.asarray(emb[start:stop], dtype=np.float32)
        assert value.shape == (1, 512)
        if label not in reduced:
            reduced[label] = (value[0] - pca["mean"]) @ pca["components"].T
        return value[0], reduced[label]

    accepted = []
    seen = {}
    for rec in candidates:
        family = rec["family"]
        reasons = []
        if rec["split"] not in ["train", "val"]:
            reasons.append("held_out_or_unassigned_source_family")
        if family in exclude:
            reasons.append("explicit_source_quarantine")
        if raw.get(family, {}).get("status") != "raw_header_valid":
            reasons.append("no_verified_amass_source")
        if reasons:
            rejected.append(dict(rec, reasons=reasons))
            continue
        full = (
            root
            / "outputs_amass/unified_direct_20261008/data/motions"
            / (family + ".npy")
        )
        if not full.exists():
            full = out / "motions" / (family + ".npy")
            status = convert_source(
                (
                    str(root / "work/unified_direct/prepare_data.py"),
                    (family, raw[family]["raw_path"], str(full), str(skpath)),
                )
            )
            if status["status"] not in ["ok", "cached"]:
                rejected.append(
                    dict(rec, reasons=["source_conversion_failed"], conversion=status)
                )
                continue
        if family not in sources:
            sources[family] = dict(
                path=str(full),
                sha256=sha(full),
                raw_path=raw[family]["raw_path"],
                raw_sha256=sha(raw[family]["raw_path"]),
            )
        motion = np.load(full, mmap_mode="r")
        if (
            "babel_duration_s" in rec
            and abs(len(motion) / 20 - rec["babel_duration_s"]) > 0.15
        ):
            rejected.append(
                dict(
                    rec,
                    reasons=["source_duration_mismatch"],
                    actual_duration=len(motion) / 20,
                )
            )
            continue
        start = max(0, int(np.ceil(rec["start_s"] * 20)))
        stop = min(len(motion), int(np.floor(rec["end_s"] * 20)))
        for start, stop in chunks(start, stop):
            item = dict(rec, crop_start_frame_20fps=start, crop_end_frame_20fps=stop)
            clip = motion[start:stop].copy()
            n = len(clip)
            with torch.no_grad():
                j = sk(torch.from_numpy(clip)[None])[0].numpy()
                q = float(
                    measure(
                        sk,
                        torch.from_numpy(clip)[None],
                        torch.tensor([3]),
                        torch.tensor([n]),
                    )[0]
                )
            values = dict(metrics(j), **waveform(j))
            reasons = []
            for name in [
                "root_excursion_m",
                "root_mean_speed_m_s",
                "foot_excursion_m",
                "root_height_range_m",
            ]:
                if values[name] > ADMISSION[name]:
                    reasons.append(name + "_above_limit")
            if (
                values["right_wrist_relative_excursion_m"]
                < ADMISSION["right_wrist_relative_excursion_m"]
            ):
                reasons.append("right_wrist_relative_excursion_m_below_limit")
            values["foot_height_range_m"] = float(
                np.ptp(j[:, [7, 8, 10, 11], 2], axis=0).max()
            )
            if values["foot_height_range_m"] > 0.12:
                reasons.append("foot_height_range_m_above_limit")
            torso = (j[:, 16] + j[:, 17]) / 2 - j[:, 0]
            values["max_torso_tilt_rad"] = float(
                np.arctan2(np.linalg.norm(torso[:, :2], axis=-1), torso[:, 2]).max()
            )
            if values["max_torso_tilt_rad"] > 0.6:
                reasons.append("deep_torso_lean_during_wave")
            if values["lateral_half_range_m"] < 0.03:
                reasons.append("no_observable_right_lateral_wave")
            if values["lateral_reversals"] < 2:
                reasons.append("no_repeated_right_lateral_wave")
            if values["right_wrist_raised_fraction"] < 0.75:
                reasons.append("right_wrist_not_sustained_raised")
            if reasons:
                rejected.append(dict(item, metrics=values, reasons=reasons))
                continue
            previous = seen.setdefault(family, [])
            duplicate = next(
                (
                    x
                    for x in previous
                    if max(0, min(stop, x[1]) - max(start, x[0]))
                    / min(stop - start, x[1] - x[0])
                    >= 0.5
                ),
                None,
            )
            if duplicate:
                rejected.append(
                    dict(
                        item,
                        reasons=["overlapping_source_event"],
                        duplicate_of=duplicate[2],
                    )
                )
                continue
            # Source event label replaces inaccurate whole-take caption. Only explicit or measured active arms are assigned.
            active_arms = ["right_arm"]
            left_lateral = np.ptp((j[:, 20] - j[:, 0])[:, 1])
            if (
                re.search(r"both|arms|hands", rec["label"])
                and left_lateral >= 0.12
                and ((j[:, 20, 2] - j[:, 0, 2]) > 0.2).mean() >= 0.75
            ):
                active_arms.append("left_arm")
            local = np.zeros((n, 408), np.float32)
            mask = np.zeros_like(local, dtype=bool)
            local_labels = {
                part: rec["label"]
                for part in parts
                if part in active_arms or part == "action"
            }
            for pi, part in enumerate(parts):
                if part in local_labels:
                    local[:, pi * 51 : (pi + 1) * 51] = embed(local_labels[part])[1]
                    mask[:, pi * 51 : (pi + 1) * 51] = True
            key = f"wave_{rec['source_kind']}_{Path(family).stem}_{start}_{stop}"
            cache = out / "cache" / f"{rec['split']}_{key}.npz"
            assert (
                np.isfinite(clip).all() and np.isfinite(q) and np.isfinite(local).all()
            )
            np.savez(
                cache,
                motion=clip,
                local=local,
                local_mask=mask,
                tx=embed(rec["label"])[0],
                quantity=np.float32(q),
                task=np.int64(3),
            )
            accepted.append(
                dict(
                    item,
                    key=key,
                    task="wave",
                    task_id=3,
                    caption=rec["label"],
                    cache=str(cache),
                    cache_sha256=sha(cache),
                    quantity=q,
                    target_frames=n,
                    real_frames=n,
                    pad_frames=0,
                    motion_source=str(full),
                    metrics=values,
                    local_labels=local_labels,
                    frankenstein_annotation_keys=frank.get(family, []),
                    ready_for_training=True,
                    manual_verified=False,
                    review_status="awaiting_user_review",
                    text_policy="Actual event source label as global/action/active-arm text; unspecified parts unknown. Original CLIP/PCA; no full-take caption and no synthetic prompt.",
                )
            )
            previous.append((start, stop, key))
    for split in ["train", "val"]:
        original = json.loads((args.base / f"{split}.json").read_text())
        keep = [x for x in original if x["task"] != "wave"]
        add = [x for x in accepted if x["split"] == split]
        assert add, "No admitted " + split + " wave"
        save_json(out / f"{split}.json", keep + add)
    save_json(out / "wave_index.json", accepted)
    save_json(out / "rejected.json", rejected)
    save_json(out / "sources.json", sources)
    provenance.update(
        {
            str(p): sha(p)
            for p in [
                args.base / "train.json",
                args.base / "val.json",
                annroot / "annotations.json",
                skpath,
                pcapath,
                official,
                nativepath,
                Path(__file__),
                Path(__file__).with_name("rules.py"),
            ]
        }
    )
    save_json(out / "provenance.json", provenance)
    policy = dict(
        ADMISSION,
        name="native_hdm05_plus_babel_act_cat_wave_v3",
        priority=["native_script_supplement", "BABEL frame act_cat=wave"],
        no_regex_only_admission=True,
        maximum_foot_height_range_m=0.12,
        maximum_torso_tilt_rad=0.6,
        minimum_lateral_half_range_m=0.03,
        minimum_lateral_reversals=2,
        minimum_raised_fraction=0.75,
        babel_allowed_categories=sorted(ALLOWED_CATEGORIES),
        babel_harmless_overlap=sorted(HARMLESS_OVERLAP),
        split_policy="Existing Frankenstein source-family split is authoritative for all20 tasks. BABEL splits describe annotation provenance only; no existing family is moved. Existing test and unassigned families excluded. BABEL test not loaded.",
        label_policy="Event text replaces whole-sequence caption. Unspecified body parts masked unknown.",
        manual_native_boundaries=native,
        threshold_basis="Training HDM05 diagnostics show pelvis sway can exceed0.12m/s while feet remain planted. Tighten foot excursion to0.15m and foot height range to0.12m; permit bounded pelvis sway. No validation examples used to set thresholds.",
    )
    save_json(out / "policy.json", policy)
    summary = dict(
        state="awaiting_user_review",
        base=str(args.base),
        counts=dict(Counter(x["split"] for x in accepted)),
        by_source=dict(Counter(x["split"] + ":" + x["source_kind"] for x in accepted)),
        families={
            s: len({x["family"] for x in accepted if x["split"] == s})
            for s in ["train", "val"]
        },
        rejection_counts=dict(Counter(y for x in rejected for y in x["reasons"])),
        command_range={
            s: [
                min(x["quantity"] for x in accepted if x["split"] == s),
                max(x["quantity"] for x in accepted if x["split"] == s),
            ]
            for s in ["train", "val"]
        },
        other19_tasks_unchanged=True,
        training_launched=False,
        test_sources_admitted=0,
        source_family_split_unchanged=True,
        source_candidates=len(candidates),
        limitations=[
            "Native scene name is not an event boundary or verification. Supplemental cuts await user review.",
            "BABEL categories can overlap or be coarse; motion admission and visual review remain necessary.",
            "Existing right-wrist metric retained; left-only waves excluded.",
        ],
    )
    save_json(out / "audit.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
