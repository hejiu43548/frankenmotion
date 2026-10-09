"""Locate real right-forward punch events; repair only strike manifests."""

import argparse
from collections import Counter
import json
from pathlib import Path
import re
import sys
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.prepare_amass20 import convert_source
from shared_motion.training.geometry import Skeleton
from shared_motion.training.catalog import measure
from shared_motion.training.model import file_sha256 as sha
from shared_motion.training.runner import save_json
from data_processing.wave.repair import canonical
from data_processing.strike.rules import (
    POLICY,
    CONTEXT_CATS,
    semantic_reasons,
    signals,
    peaks,
    admission,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument(
        "--base",
        type=Path,
        default=Path(
            "/mnt/sda2/frankenmotion/outputs_amass/source_repair_wave_20261009_review_v2"
        ),
    )
    ap.add_argument(
        "--project", type=Path, default=Path("/home/psirobot/projects/frankenmotion")
    )
    ap.add_argument("--babel", type=Path, default=Path("/mnt/sda2/dataset/babel"))
    args = ap.parse_args()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "cache").mkdir()
    torch.set_num_threads(2)
    root = args.project
    ar = root / "datasets/annotations/frankenstein-dataset/annotations"
    anns = json.loads((ar / "annotations.json").read_text())
    splits = {}
    familykeys = {}
    for s in ["train", "val", "test"]:
        for k in (ar / "splits" / f"{s}.txt").read_text().split():
            family = anns[k]["path"]
            assert splits.setdefault(family, s) == s
            familykeys.setdefault(family, []).append(k)
    rawindex = (
        root / "outputs_amass/stage2_manifests_20261008/annotated_source_raw_audit.json"
    )
    raw = json.loads(rawindex.read_text())
    skpath = (
        root
        / "outputs_amass/transfer_charlie_20261008/snapshot/outputs_amass/franken_eleven_20261003/skeleton.npz"
    )
    sk = Skeleton(skpath)
    candidates = []
    rejected = []
    byfamily = {}
    inputs = {}
    for name in ["train.json", "val.json", "extra_train.json", "extra_val.json"]:
        path = args.babel / name
        inputs[str(path)] = sha(path)
        for sid, r in json.loads(path.read_text()).items():
            family = canonical(r["feat_p"])
            blocks = (
                (r.get("frame_anns") or [])
                if name.startswith("extra")
                else [r.get("frame_ann")]
            )
            for block in blocks:
                if not block:
                    continue
                byfamily.setdefault(family, []).extend(block["labels"])
                for label in block["labels"]:
                    if "punch" not in (label.get("act_cat") or []):
                        continue
                    record = dict(
                        family=family,
                        split=splits.get(family),
                        babel_file=name,
                        babel_sid=sid,
                        babel_lid=block.get("babel_lid"),
                        seg_id=label["seg_id"],
                        label=label["proc_label"],
                        source_label=label,
                        source_kind=(
                            "hdm05_babel_punch_frame"
                            if "MPI_HDM05/" in family and "_03-02_" in family
                            else "babel_punch_frame"
                        ),
                        babel_duration_s=r["dur"],
                    )
                    reasons = semantic_reasons(label)
                    if record["split"] not in ["train", "val"]:
                        reasons.append("test_or_unassigned_source_family")
                    if raw.get(family, {}).get("status") != "raw_header_valid":
                        reasons.append("missing_verified_raw_source")
                    if reasons:
                        rejected.append(dict(record, reasons=reasons))
                        continue
                    candidates.append((record, block["labels"]))
    candidates.sort(
        key=lambda x: (
            x[0]["source_kind"] != "hdm05_babel_punch_frame",
            x[0]["family"],
            x[0]["source_label"]["start_t"],
        )
    )
    sources = {}
    loaded = {}
    admitted = []
    seen = {}
    old_audit = []

    def source(family):
        if family in loaded:
            return loaded[family]
        path = (
            root
            / "outputs_amass/unified_direct_20261008/data/motions"
            / (family + ".npy")
        )
        if not path.exists():
            path = out / "motions" / (family + ".npy")
            result = convert_source(
                (
                    str(root / "work/unified_direct/prepare_data.py"),
                    (family, raw[family]["raw_path"], str(path), str(skpath)),
                )
            )
            assert result["status"] in ["ok", "cached"], result
        motion = np.load(path)
        with torch.no_grad():
            j = sk(torch.from_numpy(motion)[None])[0].numpy()
        sources[family] = dict(
            motion_path=str(path),
            motion_sha256=sha(path),
            raw_path=raw[family]["raw_path"],
            raw_sha256=sha(raw[family]["raw_path"]),
        )
        loaded[family] = (path, motion, j)
        return loaded[family]

    for rec, labels in candidates:
        family = rec["family"]
        path, motion, j = source(family)
        if abs(len(motion) / 20 - rec["babel_duration_s"]) > 0.15:
            rejected.append(dict(rec, reasons=["babel_raw_duration_mismatch"]))
            continue
        s = int(np.ceil(rec["source_label"]["start_t"] * 20))
        e = int(np.floor(rec["source_label"]["end_t"] * 20))
        found = peaks(j, s, e)
        if not found:
            rejected.append(
                dict(rec, reasons=["no_right_forward_speed_peak_in_labeled_event"])
            )
            continue
        for peak in found:
            attempts = []
            selected = None
            for n in POLICY["crop_frame_options"]:
                for local_peak in POLICY["peak_frame_options"]:
                    start = peak - local_peak
                    stop = start + n
                    event = dict(
                        rec,
                        source_peak_frame_20fps=peak,
                        local_peak_frame=local_peak,
                        crop_start_frame_20fps=start,
                        crop_end_frame_20fps=stop,
                    )
                    if start < 0 or stop > len(motion):
                        attempts.append(
                            dict(event, reasons=["cannot_fit_unpadded_event_context"])
                        )
                        continue
                    conflict = []
                    for other in labels:
                        overlap = max(
                            0,
                            min(stop / 20, other["end_t"])
                            - max(start / 20, other["start_t"]),
                        )
                        if other["seg_id"] == rec["seg_id"] or overlap < 0.1:
                            continue
                        cats = set(other.get("act_cat") or [])
                        if cats - CONTEXT_CATS or (
                            "punch" in cats and semantic_reasons(other)
                        ):
                            conflict.append(other)
                    clip = motion[start:stop].copy()
                    with torch.no_grad():
                        cj = sk(torch.from_numpy(clip)[None])[0].numpy()
                    values, reasons = admission(cj, local_peak)
                    if conflict:
                        reasons.append("conflicting_action_in_event_context")
                    if reasons:
                        attempts.append(
                            dict(
                                event,
                                reasons=reasons,
                                metrics=values,
                                conflicting_labels=conflict,
                            )
                        )
                        continue
                    selected = (event, values, clip, start, stop, n)
                    break
                if selected is not None:
                    break
            if selected is None:
                rejected.append(
                    dict(
                        rec,
                        source_peak_frame_20fps=peak,
                        reasons=sorted({x for a in attempts for x in a["reasons"]}),
                        crop_attempts=attempts,
                    )
                )
                continue
            event, values, clip, start, stop, n = selected
            duplicates = seen.setdefault(family, [])
            prior = next(
                (
                    r
                    for r in duplicates
                    if max(0, min(stop, r[1]) - max(start, r[0])) / min(n, r[1] - r[0])
                    >= 0.5
                ),
                None,
            )
            if prior:
                rejected.append(
                    dict(
                        event,
                        reasons=["overlapping_source_event"],
                        duplicate_of=prior[2],
                    )
                )
                continue
            identifier = f'strike_{rec["babel_sid"]}_{start}_{stop}'
            with torch.no_grad():
                quantity = float(
                    measure(
                        sk,
                        torch.from_numpy(clip)[None],
                        torch.tensor([2]),
                        torch.tensor([n]),
                    )[0]
                )
            assert np.isfinite(clip).all() and np.isfinite(quantity)
            admitted.append(
                dict(
                    event,
                    key=identifier,
                    task="strike",
                    task_id=2,
                    caption=rec["label"],
                    quantity=quantity,
                    real_frames=n,
                    target_frames=n,
                    pad_frames=0,
                    motion_source=str(path),
                    metrics=values,
                    frankenstein_annotation_keys=familykeys[family],
                    ready_for_training=True,
                    manual_verified=False,
                    review_status="awaiting_user_review",
                )
            )
            duplicates.append((start, stop, identifier))
    # Reconstruct unchanged-source text: exact event text globally, timed action/right-arm only.
    er = root / "datasets/annotations/frankenstein-dataset/text_embeddings/clip"
    emb = np.load(er / "clip.npy", mmap_mode="r")
    slices = np.load(er / "clip_slice.npy")
    index = json.loads((er / "clip_index.json").read_text())
    pcapath = root / "outputs_amass/unified_direct_20261008/data/pca.npz"
    pca = np.load(pcapath)
    official = root / "pretrained/official/config.json"
    parts = json.loads(official.read_text())["data"]["text_encoder"]["body_part_order"]
    missing = sorted({r["caption"] for r in admitted if r["caption"] not in index})
    extra = {}
    parity = {}
    if missing:
        import clip

        checkpoint = Path(
            "/mnt/sda2/dataset/mdm_refine_phc_mvp_v1/assets/clip/ViT-B-32.pt"
        )
        inputs[str(checkpoint)] = sha(checkpoint)
        model, _ = clip.load(
            str(checkpoint), device="cuda" if torch.cuda.is_available() else "cpu"
        )
        model.eval()
        device = next(model.parameters()).device
        check = ["punch", "punch with right arm"]
        texts = missing + check
        with torch.no_grad():
            encoded = (
                model.encode_text(clip.tokenize(texts).to(device)).float().cpu().numpy()
            )
        for label, value in zip(texts, encoded):
            if label in missing:
                extra[label] = value
            else:
                begin, end = slices[index[label]]
                old = emb[begin:end][0]
                cosine = float(
                    np.dot(old, value) / (np.linalg.norm(old) * np.linalg.norm(value))
                )
                assert cosine > 0.9999
                parity[label] = cosine
        np.savez(
            out / "new_event_embeddings.npz",
            **{str(i): extra[label] for i, label in enumerate(missing)},
        )
        save_json(
            out / "new_event_embeddings.json",
            dict(labels=missing, parity_cosine=parity, model_sha256=sha(checkpoint)),
        )
    for r in admitted:
        label = r["caption"]
        if label in extra:
            tx = extra[label]
        else:
            begin, end = slices[index[label]]
            tx = np.asarray(emb[begin:end])[0]
        red = (tx - pca["mean"]) @ pca["components"].T
        start = r["crop_start_frame_20fps"]
        stop = r["crop_end_frame_20fps"]
        motion = np.load(r["motion_source"], mmap_mode="r")[start:stop].copy()
        local = np.zeros((stop - start, 408), np.float32)
        mask = np.zeros_like(local, bool)
        times = np.arange(start, stop) / 20
        active = (times >= r["source_label"]["start_t"]) & (
            times < r["source_label"]["end_t"]
        )
        for pi, part in enumerate(parts):
            if part in ["action", "right_arm"]:
                local[active, pi * 51 : (pi + 1) * 51] = red
                mask[active, pi * 51 : (pi + 1) * 51] = True
        cache = out / "cache" / f"{r['split']}_{r['key']}.npz"
        np.savez(
            cache,
            motion=motion,
            local=local,
            local_mask=mask,
            tx=tx,
            quantity=np.float32(r["quantity"]),
            task=np.int64(2),
        )
        r.update(
            cache=str(cache),
            cache_sha256=sha(cache),
            text_policy="Original BABEL event text; action/right_arm only during labeled source time; unlabeled preparation/reset context masked unknown. Global event text, unchanged CLIP/PCA.",
        )
    for split in ["train", "val"]:
        old = json.loads((args.base / f"{split}.json").read_text())
        for r in old:
            if r["task"] != "strike":
                continue
            clip = np.load(r["cache"])["motion"][: r["real_frames"]]
            with torch.no_grad():
                j = sk(torch.from_numpy(clip)[None])[0].numpy()
            sig = signals(j)
            mp = 14 + int(np.argmax(sig["world_speed"][14:32]))
            source_t = (r["crop_start_frame_20fps"] + mp + 2) / 20
            active = [
                l
                for l in byfamily.get(r["family"], [])
                if l["start_t"] <= source_t < l["end_t"]
            ]
            cats = sorted({c for l in active for c in (l.get("act_cat") or [])})
            reasons = []
            if re.search(r"\bbox\b", r["caption"], re.I) and not re.search(
                r"\bpunch\w*|\bjab\w*|\bboxing\b", r["caption"], re.I
            ):
                reasons.append("object_box_lexical_collision")
            if "punch" not in cats:
                reasons.append("no_babel_punch_at_old_metric_peak")
            if sig["forward_speed"][mp] < 0.8:
                reasons.append("no_fast_right_forward_punch_at_metric_peak")
            old_audit.append(
                dict(
                    split=split,
                    key=r["key"],
                    family=r["family"],
                    caption=r["caption"],
                    old_quantity=r["quantity"],
                    source_metric_peak_s=source_t,
                    active_babel_labels=active,
                    active_babel_categories=cats,
                    right_forward_speed_m_s=float(sig["forward_speed"][mp]),
                    reasons=reasons,
                )
            )
        keep = [r for r in old if r["task"] != "strike"]
        add = [r for r in admitted if r["split"] == split]
        # Empty validation is reported explicitly instead of stealing test/training sources.
        save_json(out / f"{split}.json", keep + add)
    save_json(out / "strike_index.json", admitted)
    save_json(out / "rejected.json", rejected)
    save_json(out / "old_strike_audit.json", old_audit)
    save_json(out / "sources.json", sources)
    policy = dict(
        POLICY,
        babel_allowed_categories=sorted(CONTEXT_CATS),
        split_policy="Existing Frankenstein source-family split authoritative; BABEL filename is annotation provenance. No test/unassigned family admitted.",
        native_boundary_status="Official HDM05 cuts endpoint returns403. HDM05 selected using script scene context and BABEL frame labels, not represented as verified official native cuts.",
    )
    save_json(out / "policy.json", policy)
    inputs.update(
        {
            str(p): sha(p)
            for p in [
                args.base / "train.json",
                args.base / "val.json",
                ar / "annotations.json",
                skpath,
                pcapath,
                official,
                rawindex,
                Path(__file__),
                Path(__file__).with_name("rules.py"),
            ]
        }
    )
    save_json(out / "provenance.json", inputs)
    summary = dict(
        state="awaiting_user_review",
        base=str(args.base),
        scope="strike only; right forward straight punch/jab",
        counts=dict(Counter(r["split"] for r in admitted)),
        by_source=dict(Counter(r["split"] + ":" + r["source_kind"] for r in admitted)),
        families={
            s: len({r["family"] for r in admitted if r["split"] == s})
            for s in ["train", "val"]
        },
        rejection_counts=dict(Counter(x for r in rejected for x in r["reasons"])),
        old_train_problem_flags=dict(
            Counter(x for r in old_audit if r["split"] == "train" for x in r["reasons"])
        ),
        other19_tasks_unchanged=True,
        training_launched=False,
        parameter_range={
            s: [
                min([r["quantity"] for r in admitted if r["split"] == s], default=None),
                max([r["quantity"] for r in admitted if r["split"] == s], default=None),
            ]
            for s in ["train", "val"]
        },
    )
    save_json(out / "audit.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
