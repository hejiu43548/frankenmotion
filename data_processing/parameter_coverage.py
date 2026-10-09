"""Report measured command support without treating quantity coverage as semantic QA."""

import hashlib
import json
from pathlib import Path

import hydra
import matplotlib
import numpy as np
from omegaconf import OmegaConf

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from shared_motion.training.catalog import COMMAND_RANGES
from shared_motion.training.catalog import TASK_NAMES


def file_digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize_split(records, boundaries):
    measured_values = np.array([record["quantity"] for record in records], dtype=float)
    assert np.isfinite(measured_values).all()
    bin_records = []
    for bin_index, (lower, upper) in enumerate(zip(boundaries[:-1], boundaries[1:])):
        selected = [
            record
            for record in records
            if lower <= record["quantity"]
            and (
                record["quantity"] < upper
                or (bin_index == len(boundaries) - 2 and record["quantity"] == upper)
            )
        ]
        bin_records.append(
            dict(
                lower=float(lower),
                upper=float(upper),
                samples=len(selected),
                families=len({record["family"] for record in selected}),
                keys=[record["key"] for record in selected],
            )
        )
    return dict(
        samples=len(records),
        families=len({record["family"] for record in records}),
        minimum=float(measured_values.min()) if len(records) else None,
        maximum=float(measured_values.max()) if len(records) else None,
        inside_range=int(
            (
                (measured_values >= boundaries[0]) & (measured_values <= boundaries[-1])
            ).sum()
        ),
        below_range=int((measured_values < boundaries[0]).sum()),
        above_range=int((measured_values > boundaries[-1]).sum()),
        empty_bins=[
            bin_index
            for bin_index, item in enumerate(bin_records)
            if not item["samples"]
        ],
        bins=bin_records,
    )


@hydra.main(
    version_base="1.3", config_path="../config", config_name="parameter_coverage"
)
def main(config):
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    manifests = {
        split: Path(config.dataset_directory) / f"{split}.json" for split in ["train", "val"]
    }
    records_by_split = {
        split: json.loads(path.read_text()) for split, path in manifests.items()
    }
    tasks = []
    for task_name, requested_range in zip(TASK_NAMES, COMMAND_RANGES):
        boundaries = np.linspace(*requested_range, config.bins + 1)
        tasks.append(
            dict(
                task=task_name,
                requested_range=list(requested_range),
                splits={
                    split: summarize_split(
                        [record for record in records if record["task"] == task_name],
                        boundaries,
                    )
                    for split, records in records_by_split.items()
                },
            )
        )
    report = dict(
        tasks=tasks,
        bins=config.bins,
        source_manifests={
            split: dict(path=str(path), sha256=file_digest(path))
            for split, path in manifests.items()
        },
        code_sha256=file_digest(__file__),
        limitations=[
            "Numbers use measured manifest quantities; unreviewed task semantics may be wrong. Quantity-bin counts do not certify clean supervision.",
            "Each bin counts distinct original recordings; a recording may appear in multiple bins, so family counts across bins are not additive.",
            "No test data inspected. Bin boundaries describe support, not a learned minimum sample-size guarantee.",
        ],
    )
    (output / "coverage.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    OmegaConf.save(config, output / "config.yaml", resolve=True)
    figure, axes = plt.subplots(1, 2, figsize=(15, 10), layout="constrained")
    for axis, split in zip(axes, ["train", "val"]):
        counts = np.array(
            [
                [item["families"] for item in task["splits"][split]["bins"]]
                for task in tasks
            ]
        )
        axis.imshow(np.log1p(counts), aspect="auto", cmap="YlGnBu")
        axis.set_yticks(range(len(tasks)), [task["task"] for task in tasks])
        axis.set_xticks(
            range(config.bins), [f"{index+1}" for index in range(config.bins)]
        )
        axis.set_title(f"{split}: distinct source families per command bin")
        axis.set_xlabel("Equal-width bins within each task's requested range")
        for task_index in range(len(tasks)):
            for bin_index in range(config.bins):
                axis.text(
                    bin_index,
                    task_index,
                    str(counts[task_index, bin_index]),
                    ha="center",
                    va="center",
                    fontsize=8,
                    color=(
                        "white"
                        if np.log1p(counts[task_index, bin_index]) > 3
                        else "black"
                    ),
                )
    figure.suptitle(
        "Measured parameter coverage — unreviewed semantics are not certified"
    )
    figure.savefig(output / "coverage.png", dpi=150)
    plt.close(figure)
    lines = [
        "# 参数监督覆盖审计",
        "",
        "按当前命令范围等宽分为10箱；统计样本及独立原始录制。未审核类别可能含错误动作，因此有数量也不代表监督有效。",
        "",
        "| 任务 | 目标范围 | 训练范围内/总数 | 训练空箱 | 验证范围内/总数 | 验证空箱 |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for task in tasks:
        training = task["splits"]["train"]
        validation = task["splits"]["val"]
        lines.append(
            f"| {task['task']} | {task['requested_range']} | {training['inside_range']}/{training['samples']} | {len(training['empty_bins'])}/{config.bins} | {validation['inside_range']}/{validation['samples']} | {len(validation['empty_bins'])}/{config.bins} |"
        )
    (output / "report.md").write_text("\n".join(lines) + "\n")
    print(
        json.dumps(next(task for task in tasks if task["task"] == "strike"), indent=2)
    )


if __name__ == "__main__":
    main()
