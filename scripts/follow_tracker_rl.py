"""Execute one prepared G1 reference using an exported online-RL tracker."""

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


@hydra.main(
    config_path="../config/tracker_rl", config_name="follow", version_base="1.3"
)
def main(configuration: DictConfig):
    output = Path(configuration.output)
    output.mkdir(parents=True, exist_ok=False)
    tracker = NativeTracker(OmegaConf.to_container(configuration, resolve=True))
    reference = dict(np.load(configuration.reference))
    if float(reference.get("fps", 0)) != 50:
        raise ValueError("Expected the prepared 50Hz reference")
    required = [
        "qpos",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
    ]
    if any(name not in reference for name in required):
        raise ValueError(
            "Run prepare_tracker_motion first; a raw GMR trajectory is not sufficient"
        )
    if any(not np.isfinite(reference[name]).all() for name in required):
        raise ValueError("Nonfinite reference state")
    if any(len(reference[name]) != len(reference["qpos"]) for name in required):
        raise ValueError("Reference arrays have inconsistent frame counts")
    record = {"task": configuration.task, "seed": configuration.motion_seed}
    result, states, actions = tracker.run_reference(reference, record)
    np.savez_compressed(output / "rollout.npz", states=states, actions=actions, fps=50)
    (output / "result.json").write_text(
        json.dumps(
            {
                "episode": result,
                "configuration": OmegaConf.to_container(configuration, resolve=True),
                "input_sha256": {
                    name: hashlib.sha256(
                        Path(configuration[name]).read_bytes()
                    ).hexdigest()
                    for name in ["reference", "policy", "contract", "scene"]
                },
                "initialization": "exact reference pose and velocity with soft joint limits; no mid-motion resets or external assistance",
            },
            indent=2,
        )
    )
    print(json.dumps(result), flush=True)


if __name__ == "__main__":
    main()
