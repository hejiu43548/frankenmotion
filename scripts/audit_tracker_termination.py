"""Reproduce native evaluations and inspect actual post-step terminal heights."""

import hashlib
import json
from pathlib import Path
import sys

import hydra
import numpy as np
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared_motion.rl.cpu import NativeTracker


@hydra.main(
    config_path="../config/tracker_rl", config_name="terminal_audit", version_base="1.3"
)
def main(configuration: DictConfig):
    artifacts = Path(configuration.artifacts)
    output = Path(configuration.output)
    if output.exists():
        raise FileExistsError(output)
    rows = []
    input_hashes = {}
    for experiment in configuration.evaluations:
        metrics_path = artifacts / "evaluation" / experiment / "metrics.json"
        original = json.loads(metrics_path.read_text())
        input_hashes[str(metrics_path)] = hashlib.sha256(
            metrics_path.read_bytes()
        ).hexdigest()
        tracker = NativeTracker(original["configuration"])
        split = original["configuration"]["split"]
        records = json.loads((artifacts / "motion" / split / "clips.json").read_text())[
            "records"
        ]
        for record in records:
            if record["task"] not in configuration.tasks:
                continue
            result, states, _ = tracker.run(record)
            prior = next(
                episode
                for episode in original["episodes"]
                if episode["task"] == record["task"]
                and episode["motion_seed"] == record["seed"]
            )
            if result != prior:
                raise ValueError(
                    f"Replay changed original metrics: {experiment}/{record['task']}/{record['seed']}"
                )
            reference_path = (
                artifacts / "motion" / split / f"{record['task']}_{record['seed']}.npz"
            )
            # A moved or changed corpus must not silently become the diagnostic input.
            if (
                hashlib.sha256(reference_path.read_bytes()).hexdigest()
                != record["motion_sha256"]
            ):
                raise ValueError(f"Reference hash mismatch: {reference_path}")
            with np.load(reference_path) as reference:
                target = reference["body_pos_w"][
                    len(states) - 1, tracker.end_effector_references, 2
                ]
            actual = tracker.data.xpos[tracker.end_effectors, 2]
            rows.append(
                {
                    "experiment": experiment,
                    "task": record["task"],
                    "motion_seed": record["seed"],
                    "frames": len(states),
                    "failure": result["failure"],
                    "terminal_measurement": "post-step actual vs current-index reference, identical evaluator convention",
                    "end_effectors": [
                        {
                            "body": tracker.model.body(body).name,
                            "target_z_m": float(target[index]),
                            "actual_z_m": float(actual[index]),
                            "signed_error_m": float(actual[index] - target[index]),
                        }
                        for index, body in enumerate(tracker.end_effectors)
                    ],
                }
            )
    if not rows:
        raise ValueError("No matching episodes")
    report = {
        "scope": "Existing evaluation replay; no threshold, reward, reference or policy change",
        "original_episode_metrics_exactly_reproduced": True,
        "input_metrics_sha256": input_hashes,
        "episodes": rows,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(f"TERMINATION_AUDIT_COMPLETE {len(rows)} episodes", flush=True)


if __name__ == "__main__":
    main()
