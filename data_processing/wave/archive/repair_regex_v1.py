#!/usr/bin/env python3
"""Replace only wave; recrop raw features/text and retain rejection evidence."""
import hashlib
import json
from collections import Counter
from pathlib import Path
import sys
import hydra
import numpy as np
from omegaconf import OmegaConf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_amass20 import convert_source
from shared_motion.training.wave_data import POLICY, WAVE, caption_reasons, evidence, windows, metrics, motion_reasons
from shared_motion.training.catalog import measure
from shared_motion.training.geometry import Skeleton
from shared_motion.training.model import file_sha256 as sha
from shared_motion.training.runner import save_json


@hydra.main(version_base="1.3", config_path="../config", config_name="repair_wave_data")
def main(config):
    out = Path(config.output).resolve()
    out.mkdir(parents=True, exist_ok=False)
    (out / "cache").mkdir()
    OmegaConf.save(config, out / "config.yaml", resolve=True)
    save_json(out / "policy.json", POLICY)
    torch.set_num_threads(2)
    ar = Path(config.annotations)
    anns = json.loads((ar / "annotations.json").read_text())
    splits = {s: (ar / "splits" / f"{s}.txt").read_text().split() for s in ["train", "val", "test"]}
    families = {s: {anns[k]["path"] for k in keys} for s, keys in splits.items()}
    assert not any(families[a] & families[b] for a,b in [("train","val"),("train","test"),("val","test")])
    raw = json.loads(Path(config.raw_index).read_text())
    sk = Skeleton(config.skeleton)
    er = Path(config.embeddings)
    emb = np.load(er / "clip.npy", mmap_mode="r")
    slices = np.load(er / "clip_slice.npy")
    idx = json.loads((er / "clip_index.json").read_text())
    pca = np.load(config.pca)
    parts = json.loads(Path(config.official_config).read_text())["data"]["text_encoder"]["body_part_order"]
    reduced = {}
    def embed(label):
        s,e = slices[idx[label]]
        value = np.asarray(emb[s:e], dtype=np.float32)
        assert value.shape == (1,512)
        if label not in reduced:
            reduced[label] = (value[0]-pca["mean"]) @ pca["components"].T
        return value[0], reduced[label]

    sources, accepted, rejected, old_audit = {}, [], [], []
    identities = {}
    def source(family):
        full = Path(config.existing_motions) / (family+".npy")
        if not full.exists():
            full = out / "motions" / (family+".npy")
            status = convert_source((config.conversion_module, (family,raw[family]["raw_path"],str(full),config.skeleton)))
            assert status["status"] in ["ok","cached"], status
        if family not in sources:
            sources[family] = dict(path=str(full), sha256=sha(full), raw_path=raw[family]["raw_path"])
        return full

    for split in ["train", "val"]:
        # Inspect old training input separately, using only its real frames.
        original = json.loads((Path(config.base) / f"{split}.json").read_text())
        for row in original:
            if row["task"] != "wave":
                continue
            ann = anns[row.get("annotation_key",row["key"])]
            with np.load(row["cache"]) as z:
                motion = z["motion"][:row["real_frames"]].copy()
            with torch.no_grad():
                values = metrics(sk(torch.from_numpy(motion)[None])[0].numpy())
            start = row["crop_start_frame_20fps"]
            mask = evidence(ann, start+len(motion))[start:]
            reasons = caption_reasons(ann) + motion_reasons(values)
            if not mask.all():
                reasons.append("old_crop_not_fully_timed_right_wave")
            old_audit.append(dict(split=split,key=row["key"],family=row["family"],caption=row["caption"],metrics=values,reasons=reasons,timed_right_wave_fraction=float(mask.mean())))
        for key in sorted(splits[split]):
            ann = anns[key]
            if not WAVE.search(ann.get("caption_label", "")) and not any(WAVE.search(x["text"]) for x in ann["annotations"]):
                continue
            family = ann["path"]
            identity = dict(split=split,annotation_key=key,family=family,caption=ann["caption_label"])
            reasons = caption_reasons(ann)
            if raw.get(family,{}).get("status") != "raw_header_valid" or "humanact12" in family.lower():
                reasons.append("missing_or_excluded_raw_source")
            if reasons:
                rejected.append(dict(identity,reasons=reasons)); continue
            full_path = source(family)
            full = np.load(full_path, mmap_mode="r")
            crops = windows(evidence(ann,len(full)))
            if not crops:
                rejected.append(dict(identity,reasons=["no_continuous_2s_timed_right_wave"])); continue
            for start, stop in crops:
                record = dict(identity,crop_start_frame_20fps=start,crop_end_frame_20fps=stop)
                motion = full[start:stop].copy()
                assert np.isfinite(motion).all()
                with torch.no_grad():
                    positions = sk(torch.from_numpy(motion)[None])[0].numpy()
                values = metrics(positions)
                reasons = motion_reasons(values)
                if reasons:
                    rejected.append(dict(record,metrics=values,reasons=reasons)); continue
                prior = identities.setdefault(family,[])
                duplicate = next((x for x in prior if max(0,min(stop,x[1])-max(start,x[0])) / min(stop-start,x[1]-x[0]) >= .5), None)
                if duplicate:
                    rejected.append(dict(record,reasons=["overlapping_source_event"],duplicate_of=duplicate[2])); continue
                n = len(motion)
                local = np.zeros((n,408),np.float32)
                mask = np.zeros_like(local,dtype=bool)
                times = np.arange(start,stop)/20
                for pi, part in enumerate(parts):
                    for fi,time in enumerate(times):
                        label = next((x["text"] for x in ann["annotations"] if x["bodypart"]==part and x["start"]<=time<x["end"]),"unknown")
                        if label != "unknown":
                            local[fi,pi*51:(pi+1)*51] = embed(label)[1]
                            mask[fi,pi*51:(pi+1)*51] = True
                with torch.no_grad():
                    q = float(measure(sk,torch.from_numpy(motion)[None],torch.tensor([3]),torch.tensor([n]))[0])
                assert np.isfinite(q) and np.isfinite(local).all()
                identifier = f"wave_{key}_{start}_{stop}"
                cache = out / "cache" / f"{split}_{identifier}.npz"
                np.savez(cache,motion=motion,local=local,local_mask=mask,tx=embed(ann["caption_label"])[0],quantity=np.float32(q),task=np.int64(3))
                cropped = [dict(x,start=max(start/20,x["start"])-start/20,end=min(stop/20,x["end"])-start/20) for x in ann["annotations"] if x["start"]<stop/20 and x["end"]>start/20]
                accepted.append(dict(record,key=identifier,task="wave",task_id=3,cache=str(cache),cache_sha256=sha(cache),quantity=q,target_frames=n,real_frames=n,pad_frames=0,motion_source=str(full_path),metrics=values,cropped_annotations=cropped,ready_for_training=True,manual_verified=False,review_status="awaiting_user_review",semantic_event_verification=POLICY["name"],text_policy="Original conflict-free caption; original timed body-part labels recropped; original PCA retained."))
                prior.append((start,stop,identifier))
        print(split, Counter(x["split"] for x in accepted), flush=True)
        keep = [r for r in original if r["task"]!="wave"]
        additions = [r for r in accepted if r["split"]==split]
        assert additions, f"No accepted {split} wave sources"
        save_json(out / f"{split}.json", keep+additions)
    save_json(out / "wave_index.json",accepted)
    save_json(out / "rejected.json",rejected)
    save_json(out / "old_wave_audit.json",old_audit)
    save_json(out / "sources.json",sources)
    summary = dict(state="awaiting_user_review",scope="wave only",training_launched=False,other_tasks_unchanged=True,split_family_isolation=True,test_used=False,sample_cap=None,counts=dict(Counter(x["split"] for x in accepted)),families={s:len({r["family"] for r in accepted if r["split"]==s}) for s in ["train","val"]},rejection_counts=dict(Counter(y for x in rejected for y in x["reasons"])),old_wave_counts=dict(Counter(x["split"] for x in old_audit)),old_train_reasons=dict(Counter(y for x in old_audit if x["split"]=="train" for y in x["reasons"])),inputs={str(p):sha(p) for p in [Path(config.base)/"train.json",Path(config.base)/"val.json",ar/"annotations.json",Path(config.pca),Path(config.skeleton),Path(config.official_config),Path(__file__),Path(__file__).resolve().parents[1]/"shared_motion/training/wave_data.py"]})
    save_json(out / "audit.json",summary)
    print(json.dumps(summary,indent=2),flush=True)


if __name__ == "__main__":
    main()
