"""Evaluate archived and newly trained policies in one native MuJoCo scene."""

from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
from pathlib import Path
import sys

import hydra
import numpy as np
from omegaconf import DictConfig
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared_motion.rl.cpu import NativeTracker

TRACKER = None
OUTPUT = None


def initialize(configuration):
    global TRACKER
    global OUTPUT
    TRACKER = NativeTracker(configuration)
    OUTPUT = Path(configuration["output"])


def evaluate(record):
    result, states, actions = TRACKER.run(record)
    np.savez_compressed(
        OUTPUT / f'{record["task"]}_{record["seed"]}.npz',
        states=states,
        actions=actions,
        fps=50,
    )
    return result


@hydra.main(
    config_path="../config/tracker_rl", config_name="cpu_evaluate", version_base="1.3"
)
def main(configuration: DictConfig):
    output = Path(configuration.output)
    output.mkdir(parents=True, exist_ok=False)
    records = json.loads(
        (
            Path(configuration.artifacts)
            / "motion"
            / configuration.split
            / "clips.json"
        ).read_text()
    )["records"]
    results = []
    with ProcessPoolExecutor(
        configuration.workers,
        initializer=initialize,
        initargs=(OmegaConf.to_container(configuration, resolve=True),),
    ) as pool:
        for result in pool.map(evaluate, records):
            results.append(result)
            (output / "episodes.json").write_text(json.dumps(results, indent=2))
            print(len(results), result["task"], result["complete"], flush=True)
    tasks = sorted({result["task"] for result in results})
    fields = [
        "complete",
        "tracking_success",
        "root_m",
        "body_m",
        "joint_rad",
        "root_m_failure_penalized",
        "body_m_failure_penalized",
        "joint_rad_failure_penalized",
        "action_delta_squared",
    ]
    per_task = {
        task: {
            field: float(
                np.mean([result[field] for result in results if result["task"] == task])
            )
            for field in fields
        }
        for task in tasks
    }
    summary = {
        field: float(np.mean([values[field] for values in per_task.values()]))
        for field in fields
    }
    (output / "metrics.json").write_text(
        json.dumps(
            {
                "configuration": OmegaConf.to_container(configuration, resolve=True),
                "policy_sha256": hashlib.sha256(
                    Path(configuration.policy).read_bytes()
                ).hexdigest(),
                "scene_sha256": hashlib.sha256(
                    Path(configuration.scene).read_bytes()
                ).hexdigest(),
                "macro": summary,
                "per_task": per_task,
                "episodes": results,
                "backend": "native MuJoCo; compare only with runs of this evaluator, separately from Warp",
            },
            indent=2,
        )
    )
    print("CPU_EVALUATION_COMPLETE", json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
