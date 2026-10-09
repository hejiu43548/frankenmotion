"""Paired task-macro comparisons with all failures retained and cluster intervals."""

import csv
import hashlib
import json
from pathlib import Path

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from omegaconf import DictConfig


FIELDS = (
    "complete",
    "tracking_success",
    "root_m_failure_penalized",
    "body_m_failure_penalized",
    "joint_rad_failure_penalized",
    "horizon_fraction",
    "action_delta_squared",
)


def read_results(path):
    result = json.loads(Path(path).read_text())
    episodes = {}
    for row in result["episodes"]:
        key = (row["task"], row["motion_seed"])
        if key in episodes:
            raise ValueError(f"Duplicate episode key: {key}")
        if not 0 < row["frames"] <= row["expected_frames"]:
            raise ValueError(f"Invalid episode horizon: {key}")
        row["horizon_fraction"] = row["frames"] / row["expected_frames"]
        if not all(np.isfinite(float(row[field])) for field in FIELDS):
            raise ValueError(f"Nonfinite metric: {key}")
        episodes[key] = row
    return result, episodes


@hydra.main(
    config_path="../config/tracker_rl", config_name="compare", version_base="1.3"
)
def main(configuration: DictConfig):
    baseline, baseline_episodes = read_results(configuration.baseline)
    candidate, candidate_episodes = read_results(configuration.candidate)
    if baseline_episodes.keys() != candidate_episodes.keys():
        raise ValueError("Motion identities differ; cannot make a paired comparison")
    for field in ["backend", "scene_sha256", "references"]:
        if field not in baseline or baseline[field] != candidate.get(field):
            raise ValueError(f"Evaluation contract mismatch: {field}")
    for field in ["split", "seed", "perturbation"]:
        if baseline["configuration"][field] != candidate["configuration"][field]:
            raise ValueError(f"Evaluation conditions differ: {field}")
    for key in baseline_episodes:
        if (
            baseline_episodes[key]["expected_frames"]
            != candidate_episodes[key]["expected_frames"]
        ):
            raise ValueError(f"Reference horizon differs: {key}")
    output = Path(configuration.output)
    output.mkdir(parents=True, exist_ok=False)
    tasks = sorted({key[0] for key in baseline_episodes})
    task_rows = []
    for task in tasks:
        keys = [key for key in baseline_episodes if key[0] == task]
        row = {"task": task, "clips": len(keys)}
        for field in FIELDS:
            for label, episodes in [
                ("baseline", baseline_episodes),
                ("candidate", candidate_episodes),
            ]:
                row[f"{label}_{field}"] = float(
                    np.mean([episodes[key][field] for key in keys])
                )
            row[f"delta_{field}"] = row[f"candidate_{field}"] - row[f"baseline_{field}"]
        task_rows.append(row)
    random = np.random.default_rng(configuration.seed)
    sampled_tasks = random.integers(
        len(tasks), size=(configuration.bootstrap_samples, len(tasks))
    )
    summary = {}
    for field in FIELDS:
        differences = np.asarray([row[f"delta_{field}"] for row in task_rows])
        replicates = differences[sampled_tasks].mean(axis=1)
        summary[field] = {
            "baseline": float(np.mean([row[f"baseline_{field}"] for row in task_rows])),
            "candidate": float(
                np.mean([row[f"candidate_{field}"] for row in task_rows])
            ),
            "candidate_minus_baseline": float(differences.mean()),
            "task_cluster_bootstrap_95_percent_interval": np.quantile(
                replicates, [0.025, 0.975]
            ).tolist(),
        }
    result = {
        "summary": summary,
        "tasks": len(tasks),
        "clips": len(baseline_episodes),
        "input_sha256": {
            name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
            for name, path in [
                ("baseline", configuration.baseline),
                ("candidate", configuration.candidate),
            ]
        },
        "interval_interpretation": "Paired resampling of observed task clusters, retaining all their motion seeds. Describes variability across this small task mix, not training-seed uncertainty or proof of novel-task generalization. Selection-set intervals are exploratory.",
        "delta_direction": "Positive favors candidate for success/completion/horizon; negative favors candidate for errors. Lower action variation alone does not establish better tracking.",
    }
    (output / "comparison.json").write_text(json.dumps(result, indent=2))
    with (output / "per_task.csv").open("w") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(task_rows[0]))
        writer.writeheader()
        writer.writerows(task_rows)
    figure, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    positions = np.arange(len(tasks))
    for offset, label, title in [
        (-0.2, "baseline", configuration.baseline_label),
        (0.2, "candidate", configuration.candidate_label),
    ]:
        axes[0].bar(
            positions + offset,
            [row[f"{label}_tracking_success"] for row in task_rows],
            width=0.4,
            label=title,
        )
    axes[0].set(
        xticks=positions,
        xticklabels=tasks,
        ylabel="Strict tracking success",
        ylim=(0, 1.05),
    )
    axes[0].tick_params(axis="x", labelrotation=90)
    axes[0].legend(loc="lower left", bbox_to_anchor=(0, 1.02), ncol=2, frameon=False)
    timeline = np.linspace(0, 1, 101)
    for title, episodes in [
        (configuration.baseline_label, baseline_episodes),
        (configuration.candidate_label, candidate_episodes),
    ]:
        survival = [
            np.mean(
                [
                    np.mean(
                        [
                            row["horizon_fraction"] >= moment
                            for key, row in episodes.items()
                            if key[0] == task
                        ]
                    )
                    for task in tasks
                ]
            )
            for moment in timeline
        ]
        axes[1].plot(timeline, survival, label=title)
    axes[1].set(
        xlabel="Fraction of reference horizon observed",
        ylabel="Task-macro fraction reaching this point",
        ylim=(0, 1.05),
    )
    axes[1].legend()
    figure.savefig(output / "comparison.png", dpi=160)
    plt.close(figure)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
