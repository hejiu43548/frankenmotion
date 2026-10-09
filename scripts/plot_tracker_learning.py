"""Plot validation curves against actual simulator transition budgets."""

import json
from pathlib import Path

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from omegaconf import DictConfig


@hydra.main(
    config_path="../config/tracker_rl", config_name="learning_plot", version_base="1.3"
)
def main(configuration: DictConfig):
    root = Path(configuration.run_root)
    series = {}
    for filename in ["validation_curve.json", "validation_residual_curve.json"]:
        if not (root / filename).exists():
            continue
        for row in json.loads((root / filename).read_text()):
            # Stored checkpoint index is zero-based and follows its update.
            transitions = (row["iteration"] + 1) * 400 * 24
            series.setdefault(row["experiment"], []).append(
                (transitions, row["metrics"])
            )
    if (root / "validation_extension_curve.json").exists():
        selection = json.loads((root / "extension_selection.json").read_text())
        parent_iteration = selection["selected"]["iteration"]
        parent_transitions = (parent_iteration + 1) * 400 * 24
        for row in json.loads((root / "validation_extension_curve.json").read_text()):
            transitions = (
                parent_transitions
                + (row["iteration"] - parent_iteration)
                * selection["new_environments"]
                * 24
            )
            series.setdefault("selected_extension_v1", []).append(
                (transitions, row["metrics"])
            )
    if not series:
        raise ValueError("No completed validation checkpoints")
    output = Path(configuration.output)
    output.mkdir(parents=True, exist_ok=False)
    figure, axes = plt.subplots(1, 3, figsize=(15, 4.5), constrained_layout=True)
    fields = [
        ("complete", "Completion"),
        ("tracking_success", "Strict tracking success"),
        ("root_m_failure_penalized", "Root error including failure tail (m)"),
    ]
    names = {
        "baseline_current_v2": "Current reference",
        "preview_future_v2": "Future reference",
        "reference_residual_v1": "Reference + residual",
        "selected_extension_v1": "Selected policy: larger training batch",
    }
    for experiment, records in series.items():
        records.sort(key=lambda record: record[0])
        for axis, (field, title) in zip(axes, fields):
            axis.plot(
                [transitions / 1e6 for transitions, _ in records],
                [metrics[field] for _, metrics in records],
                marker="o",
                markersize=4,
                label=names.get(experiment, experiment),
            )
            axis.set(xlabel="Simulator transitions (millions)", ylabel=title)
            axis.grid(alpha=0.2)
    axes[0].set_ylim(0, 1.02)
    axes[1].set_ylim(0, 1.02)
    axes[0].legend(loc="lower right", fontsize=8)
    figure.suptitle("Validation only; deterministic full-start tracking; Warp backend")
    figure.savefig(output / "learning_curves.png", dpi=170)
    plt.close(figure)
    (output / "plot_data.json").write_text(json.dumps(series, indent=2))


if __name__ == "__main__":
    main()
