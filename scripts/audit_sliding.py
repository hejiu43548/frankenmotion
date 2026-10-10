"""Compare conservative low-foot sliding proxies on saved motion and real AMASS."""

import json
from pathlib import Path
import sys

import hydra
import matplotlib
import numpy as np
from omegaconf import OmegaConf
import torch

matplotlib.use("Agg")
from matplotlib import pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared_motion.training.geometry import Skeleton
from shared_motion.training.model import file_sha256


def continuous_segments(mask, minimum=3):
    edges = np.diff(np.concatenate([[False], mask, [False]]).astype(int))
    return [
        (start, end)
        for start, end in zip(np.flatnonzero(edges == 1), np.flatnonzero(edges == -1))
        if end - start >= minimum
    ]


def measure_sliding(positions, height_tolerance=0.05):
    feet = positions[:, [[7, 10], [8, 11]]]
    velocity = np.diff(feet, axis=0) * 20
    horizontal = np.linalg.norm(velocity[..., :2], axis=-1)
    floor = np.quantile(feet[..., 2], 0.02)
    candidates = (feet[1:, ..., 2] <= floor + height_tolerance) & (
        np.abs(velocity[..., 2]) < 0.20
    )
    support = candidates.any(axis=-1)
    segments = []
    for foot_index in range(2):
        sustained = np.zeros(len(support), dtype=bool)
        for start, end in continuous_segments(support[:, foot_index]):
            sustained[start:end] = True
            displacement = feet[end, foot_index, :, :2] - feet[start, foot_index, :, :2]
            segments.append(float(np.linalg.norm(displacement, axis=-1).min()))
        support[:, foot_index] = sustained
    conservative_speed = np.where(candidates, horizontal, np.inf).min(axis=-1)
    selected = conservative_speed[support]
    root_velocity = np.diff(positions[:, 0, :2], axis=0) * 20
    root_speed = np.linalg.norm(root_velocity, axis=-1)
    moving = root_speed > 0.15
    minimum_foot_speed = horizontal.min(axis=(1, 2))
    return dict(
        frames=len(positions),
        support_samples=int(support.sum()),
        support_fraction=float(support.mean()),
        support_speed_mean=float(selected.mean()) if len(selected) else None,
        support_speed_p95=float(np.quantile(selected, 0.95)) if len(selected) else None,
        support_slip_fraction_005=(
            float((selected > 0.05).mean()) if len(selected) else None
        ),
        support_slip_fraction_010=(
            float((selected > 0.10).mean()) if len(selected) else None
        ),
        support_slip_fraction_020=(
            float((selected > 0.20).mean()) if len(selected) else None
        ),
        support_drift_median=float(np.median(segments)) if segments else None,
        support_drift_max=max(segments) if segments else None,
        support_segments=len(segments),
        root_speed_mean=float(root_speed.mean()),
        root_path=float(root_speed.sum() / 20),
        root_net=float(np.linalg.norm(positions[-1, 0, :2] - positions[0, 0, :2])),
        moving_no_slow_foot_fraction=(
            float((minimum_foot_speed[moving] > 0.1).mean()) if moving.any() else None
        ),
        moving_min_foot_speed_ratio=(
            float(np.median(minimum_foot_speed[moving] / root_speed[moving]))
            if moving.any()
            else None
        ),
    )


def summarize(records):
    groups = {}
    for record in records:
        groups.setdefault((record["source"], record["task"]), []).append(record)
    result = []
    metrics = [
        "support_speed_mean",
        "support_speed_p95",
        "support_slip_fraction_005",
        "support_slip_fraction_010",
        "support_slip_fraction_020",
        "support_drift_median",
        "support_drift_max",
        "root_speed_mean",
        "root_path",
        "root_net",
        "moving_no_slow_foot_fraction",
        "moving_min_foot_speed_ratio",
        "support_fraction",
    ]
    for (source, task), rows in groups.items():
        entry = dict(
            source=source,
            task=task,
            clips=len(rows),
            clips_with_support=sum(row["support_samples"] > 0 for row in rows),
        )
        for name in metrics:
            values = [row[name] for row in rows if row[name] is not None]
            entry[name] = float(np.median(values)) if values else None
        result.append(entry)
    return result


@hydra.main(version_base="1.3", config_path="../config", config_name="audit_sliding")
def main(config):
    torch.set_num_threads(4)
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    skeleton = Skeleton(config.skeleton)
    run = Path(config.run)
    rows = json.loads((run / "reports/stage3/data_index/val.json").read_text())
    selected_scan = json.loads(
        (run / "reports/stage3/scans/scan_007700.json").read_text()
    )
    selected_identities = {
        (entry["task"], key, family)
        for entry in selected_scan["tasks"]
        for key, family in zip(entry["keys"], entry["families"])
    }
    records = []
    sensitivity = []
    provenance = []
    sources = [("root_only", 2, 0), ("stage2_50000", 2, 50000)] + [
        (f"stage3_{step}", 3, step) for step in range(1100, 8801, 1100)
    ]
    for source, stage, step in sources:
        scan_path = run / f"reports/stage{stage}/scans/scan_{step:06d}.json"
        scan = json.loads(scan_path.read_text())
        assert file_sha256(scan["motion_archive"]) == scan["motion_sha256"]
        provenance.append(
            dict(
                source=source, scan=str(scan_path), motion_sha256=scan["motion_sha256"]
            )
        )
        with np.load(scan["motion_archive"]) as archive:
            motion = torch.tensor(archive["motion"])
            lengths = archive["lengths"]
            task_indices = archive["task"]
        with torch.no_grad():
            positions = skeleton(motion).numpy()
        for index in range(len(motion)):
            task = scan["tasks"][int(task_indices[index])]
            points = np.flatnonzero(task_indices == task_indices[index])
            point = int(np.flatnonzero(points == index)[0])
            identity = dict(
                source=source,
                task=task["task"],
                point=point,
                key=task["keys"][point],
                family=task["families"][point],
                requested=task["requested"][point],
            )
            record = positions[index, : int(lengths[index])]
            records.append(dict(identity, **measure_sliding(record)))
            if source in ["root_only", "stage2_50000", "stage3_7700"]:
                for tolerance in [0.03, 0.05, 0.08]:
                    sensitivity.append(
                        dict(
                            identity,
                            height_tolerance=tolerance,
                            **measure_sliding(record, tolerance),
                        )
                    )
        print(source, flush=True)
    padded = 0
    for start in range(0, len(rows), 32):
        selected_rows = rows[start : start + 32]
        motions = []
        real_lengths = []
        for row in selected_rows:
            with np.load(row["cache"]) as archive:
                motion = np.asarray(archive["motion"], dtype=np.float32)
            real_frames = min(len(motion), int(row["real_frames"]))
            assert real_frames >= 3
            real_lengths.append(real_frames)
            motions.append(
                torch.from_numpy(
                    np.pad(motion[:real_frames], ((0, 120 - real_frames), (0, 0)))
                )
            )
            padded += row.get("pad_frames", 0) > 0
        with torch.no_grad():
            positions = skeleton(torch.stack(motions)).numpy()
        for index, row in enumerate(selected_rows):
            identity = dict(
                task=row["task"],
                key=row["key"],
                family=row["family"],
                real_frames=real_lengths[index],
                cache=row["cache"],
            )
            measured = measure_sliding(positions[index, : real_lengths[index]])
            records.append(dict(identity, source="amass_val_real", **measured))
            if (row["task"], row["key"], row["family"]) in selected_identities:
                records.append(dict(identity, source="amass_matched", **measured))
    summary = summarize(records)
    result = dict(
        protocol=dict(
            fps=20,
            foot_markers=[[7, 10], [8, 11]],
            floor="2nd percentile of four foot marker heights over clip",
            contact_candidate="marker <= floor+0.05m AND abs vertical velocity<0.20m/s, sustained >=3 transitions (0.15s)",
            conservative_speed="minimum horizontal speed among candidate ankle/toe markers per foot (allows foot rolling)",
            sliding_thresholds_m_s=[0.05, 0.1, 0.2],
            aggregation="task medians over per-clip values; no-contact clips excluded from contact-conditioned metrics",
            caveat="Heuristic contact candidates, not ground-truth foot contacts or validated pass/fail rates. Flat-floor assumption; no mesh sole, terrain or force data.",
            gt_padding_excluded_clips=padded,
            gt_count=len(rows),
        ),
        provenance=provenance,
        summary=summary,
        records=records,
        sensitivity=sensitivity,
    )
    (output / "sliding_metrics.json").write_text(json.dumps(result, indent=2))
    OmegaConf.save(config, output / "config.yaml", resolve=True)
    (output / "source_sha256.json").write_text(
        json.dumps(
            {
                "analysis": file_sha256(__file__),
                "skeleton": file_sha256(config.skeleton),
            },
            indent=2,
        )
    )
    tasks = [entry["task"] for entry in selected_scan["tasks"]]
    colors = ["#6d7a86", "#9467bd", "#e8a23a", "#c94444"]
    plotted_sources = ["amass_val_real", "root_only", "stage2_50000", "stage3_7700"]
    figure, axes = plt.subplots(2, 1, figsize=(16, 10), constrained_layout=True)
    indices = np.arange(len(tasks))
    for offset, (source, color) in enumerate(zip(plotted_sources, colors)):
        table = {entry["task"]: entry for entry in summary if entry["source"] == source}
        for axis, metric, label in [
            (axes[0], "support_speed_mean", "Estimated support foot speed (m/s)"),
            (
                axes[1],
                "support_slip_fraction_010",
                "Estimated support samples above0.1m/s (%)",
            ),
        ]:
            values = [
                table[task][metric] if table[task][metric] is not None else np.nan
                for task in tasks
            ]
            if metric.endswith("010"):
                values = np.asarray(values) * 100
            axis.bar(
                indices + (offset - 1.5) * 0.2,
                values,
                width=0.2,
                color=color,
                label=source,
            )
            axis.set_ylabel(label)
            axis.grid(axis="y", alpha=0.2)
            axis.set_xticks(indices, tasks, rotation=45, ha="right")
    axes[0].axhline(0.1, color="black", ls="--", lw=1)
    axes[0].legend(ncol=4)
    figure.suptitle(
        "Foot sliding diagnostic | task median over clips | heuristic support, NOT contact ground truth"
    )
    figure.savefig(output / "sliding_comparison.png", dpi=150)
    plt.close(figure)
    print(
        json.dumps(
            dict(output=str(output), records=len(records), gt_padded_excluded=padded)
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
