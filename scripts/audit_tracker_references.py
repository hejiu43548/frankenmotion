"""Audit kinematic references without inspecting policy performance or filtering."""

import hydra
from omegaconf import DictConfig
import json
from pathlib import Path
import mujoco
import numpy as np


@hydra.main(config_path="../config/tracker_rl", config_name="audit", version_base="1.3")
def main(configuration: DictConfig):
    root = Path(configuration.artifacts)
    model = mujoco.MjModel.from_binary_path(str(root / "scene/scene.mjb"))
    data = mujoco.MjData(model)
    foot_ids = [
        index
        for index in range(model.ngeom)
        if "foot" in model.geom(index).name
        and model.geom_type[index] == mujoco.mjtGeom.mjGEOM_CAPSULE
    ]
    records = []
    for split in configuration.splits:
        manifest = json.loads((root / "motion" / split / "clips.json").read_text())[
            "records"
        ]
        for record in manifest:
            motion = np.load(root / "motion" / split / Path(record["motion_path"]).name)
            minima = []
            for state in motion["qpos"]:
                data.qpos[:] = state
                mujoco.mj_forward(model, data)
                bottom = [
                    data.geom_xpos[index, 2]
                    - abs(data.geom_xmat[index].reshape(3, 3)[2, 2])
                    * model.geom_size[index, 1]
                    - model.geom_size[index, 0]
                    for index in foot_ids
                ]
                minima.append(min(bottom))
            records.append(
                {
                    "task": record["task"],
                    "seed": record["seed"],
                    "split": split,
                    "minimum_foot_height_m": float(min(minima)),
                    "penetration_fraction_below_minus_2cm": float(
                        np.mean(np.asarray(minima) < -0.02)
                    ),
                    "both_feet_above_5cm_fraction": float(
                        np.mean(np.asarray(minima) > 0.05)
                    ),
                    "max_joint_speed_rad_s": float(abs(motion["joint_vel"]).max()),
                }
            )
    (root / "reference_quality_audit.json").write_text(json.dumps(records, indent=2))
    print(
        "REFERENCE_AUDIT",
        len(records),
        "clips",
        sum(record["penetration_fraction_below_minus_2cm"] > 0.1 for record in records),
        "clips with >10% penetrated frames",
    )


if __name__ == "__main__":
    main()
