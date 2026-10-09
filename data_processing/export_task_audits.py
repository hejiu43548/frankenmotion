"""Export every task's exact input records plus human-readable audit lists."""

import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import html
import json
from pathlib import Path


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--review-state", type=Path, required=True)
    ap.add_argument(
        "--raw-index",
        type=Path,
        default=Path(
            "/home/psirobot/projects/frankenmotion/outputs_amass/stage2_manifests_20261008/annotated_source_raw_audit.json"
        ),
    )
    a = ap.parse_args()
    a.out.mkdir(parents=True, exist_ok=False)
    state = json.loads(a.review_state.read_text())
    raw_index = json.loads(a.raw_index.read_text()) if a.raw_index.is_file() else {}
    manifests = {
        s: json.loads((a.data / f"{s}.json").read_text()) for s in ["train", "val"]
    }
    cachepaths = sorted({r["cache"] for rows in manifests.values() for r in rows})

    def cacheinfo(p):
        path = Path(p)
        return (
            p,
            dict(
                exists=path.is_file(),
                bytes=path.stat().st_size if path.is_file() else None,
                sha256=sha(path) if path.is_file() else None,
            ),
        )

    with ThreadPoolExecutor(8) as pool:
        caches = dict(pool.map(cacheinfo, cachepaths))
    if not all(v["exists"] for v in caches.values()):
        raise ValueError("Missing input caches; inspect cache index")
    (a.out / "cache_index.json").write_text(json.dumps(caches, indent=2) + "\n")
    tasks = sorted({r["task"] for rows in manifests.values() for r in rows})
    assert len(tasks) == 20
    headers = [
        "task",
        "split",
        "key",
        "family",
        "source_kind",
        "source_label",
        "annotation_key",
        "source_start_s",
        "source_end_s",
        "real_frames",
        "pad_frames",
        "quantity",
        "quantity_unit",
        "babel_file",
        "babel_sid",
        "seg_id",
        "original_event_label_json",
        "motion_source",
        "task_review_status",
        "sample_manual_verified",
        "cache_sha256",
        "cache",
        "raw_path",
        "source_manifest_sha256",
    ]
    summary = []
    for task in tasks:
        folder = a.out / task
        folder.mkdir()
        counts = {}
        formatted = []
        status = state.get("tasks", {}).get(task, {}).get("status", "not_reviewed")
        for split, rows in manifests.items():
            selected = [r for r in rows if r["task"] == task]
            counts[split] = len(selected)
            (folder / f"{split}.json").write_text(
                json.dumps(selected, ensure_ascii=False, indent=2) + "\n"
            )
            digest = sha(a.data / f"{split}.json")
            for r in selected:
                s = r.get("crop_start_frame_20fps")
                n = r.get("real_frames", r.get("target_frames"))
                e = r.get(
                    "crop_end_frame_20fps",
                    (s + n) if s is not None and n is not None else None,
                )
                cache = caches[r["cache"]]
                if r.get("cache_sha256"):
                    assert r["cache_sha256"] == cache["sha256"]
                kind = r.get(
                    "source_kind",
                    r.get("semantic_event_verification", "legacy_caption_selection"),
                )
                formatted.append(
                    dict(
                        task=task,
                        split=split,
                        key=r["key"],
                        family=r["family"],
                        source_kind=kind,
                        source_label=r.get("caption", ""),
                        annotation_key=r.get("annotation_key", r["key"]),
                        source_start_s=s / 20 if s is not None else None,
                        source_end_s=e / 20 if e is not None else None,
                        real_frames=n,
                        pad_frames=r.get("pad_frames", 0),
                        quantity=r.get("quantity"),
                        quantity_unit=(
                            "rad"
                            if task in ["turn", "lean", "bow", "twist"]
                            else (
                                "m/s"
                                if task in ["strike", "back_walk", "walk", "jog"]
                                else "m"
                            )
                        ),
                        babel_file=r.get("babel_file", ""),
                        babel_sid=r.get("babel_sid", ""),
                        seg_id=r.get("seg_id", ""),
                        original_event_label_json=json.dumps(
                            r.get("source_label", {}), ensure_ascii=False
                        ),
                        motion_source=r.get("motion_source", r.get("feature_path", "")),
                        task_review_status=status,
                        sample_manual_verified=r.get("manual_verified", False),
                        cache_sha256=cache["sha256"],
                        cache=r["cache"],
                        raw_path=r.get(
                            "raw_path",
                            raw_index.get(r["family"], {}).get("raw_path", ""),
                        ),
                        source_manifest_sha256=digest,
                    )
                )
        for split in ["train", "val"]:
            with (folder / f"{split}.csv").open(
                "w", encoding="utf-8-sig", newline=""
            ) as f:
                w = csv.DictWriter(f, fieldnames=headers)
                w.writeheader()
                w.writerows(r for r in formatted if r["split"] == split)
        columns = [
            "split",
            "key",
            "family",
            "source_label",
            "source_start_s",
            "source_end_s",
            "quantity",
            "quantity_unit",
            "pad_frames",
            "task_review_status",
        ]
        table = (
            "<table><thead><tr>"
            + "".join("<th>" + c + "</th>" for c in columns)
            + "</tr></thead><tbody>"
        )
        for r in formatted:
            table += (
                "<tr>"
                + "".join("<td>" + html.escape(str(r[c])) + "</td>" for c in columns)
                + "</tr>"
            )
        page = (
            f'<!doctype html><meta charset="utf-8"><title>{task} 数据审计</title><style>body{{font:14px system-ui;margin:24px;background:#f8fafc}}table{{border-collapse:collapse;width:100%;background:white}}td,th{{border:1px solid #ddd;padding:7px;text-align:left}}td{{overflow-wrap:anywhere}}th{{position:sticky;top:0;background:#e6edf4}}</style><h1>{task} · {status}</h1><p>train={counts["train"]} / val={counts["val"]}。列表导出不等于语义审核通过；补帧单独记录。</p><p><a href="../index.html">全部任务</a> · <a href="train.csv">训练 CSV</a> · <a href="val.csv">验证 CSV</a> · <a href="train.json">完整训练记录</a> · <a href="val.json">完整验证记录</a></p>'
            + table
            + "</tbody></table>"
        )
        (folder / "index.html").write_text(page)
        summary.append(
            dict(
                task=task,
                counts=counts,
                review_status=status,
                families={
                    s: len({r["family"] for r in formatted if r["split"] == s})
                    for s in ["train", "val"]
                },
                source_kinds=dict(Counter(r["source_kind"] for r in formatted)),
                padding_records=sum(r["pad_frames"] > 0 for r in formatted),
            )
        )
    report = dict(
        data=str(a.data),
        tasks=summary,
        manifest_sha256={s: sha(a.data / f"{s}.json") for s in manifests},
        script_sha256=sha(__file__),
        review_state=state,
        all20_tasks_exported=True,
        total_rows=sum(map(len, manifests.values())),
        unique_caches_verified=len(caches),
        purpose="Exact audit lists; not semantic certification or training launch.",
    )
    (a.out / "summary.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    links = "".join(
        f'<tr><td><a href="{r["task"]}/index.html">{r["task"]}</a></td><td>{r["counts"]["train"]}</td><td>{r["counts"]["val"]}</td><td>{r["review_status"]}</td></tr>'
        for r in summary
    )
    (a.out / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>20任务训练数据审计</title><style>body{font:16px system-ui;max-width:1050px;margin:40px auto}td,th{padding:10px 24px;border-bottom:1px solid #ddd;text-align:left}</style><h1>20 个任务 · 训练数据审计</h1><p>每类独立列出训练/验证来源、时间区间、原始标签、幅度/速度、补帧、缓存哈希及审核状态。not_reviewed 表示尚未修复验收；provisionally_accepted 表示用户暂定通过。</p><table><tr><th>任务</th><th>训练</th><th>验证</th><th>审核状态</th></tr>'
        + links
        + "</table>"
    )
    print(
        json.dumps(
            dict(
                output=str(a.out),
                total_rows=report["total_rows"],
                tasks=len(tasks),
                cache_hashes=len(caches),
            )
        )
    )


if __name__ == "__main__":
    main()
