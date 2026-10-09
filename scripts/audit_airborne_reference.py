"""Diagnostic only: airborne COM acceleration of unmodified kinematic references."""

import hashlib
import json
from pathlib import Path

import hydra
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from omegaconf import DictConfig
from scipy.signal import savgol_filter


@hydra.main(
    config_path="../config/tracker_rl", config_name="airborne_audit", version_base="1.3"
)
def main(configuration: DictConfig):
    root = Path(configuration.artifacts)
    output = Path(configuration.output)
    output.mkdir(parents=True, exist_ok=False)
    scene = root / "scene/scene.mjb"
    model = mujoco.MjModel.from_binary_path(str(scene))
    data = mujoco.MjData(model)
    robot_root = model.body("robot/pelvis").id
    assert not np.any(model.body_gravcomp)
    assert model.neq == 0 and model.nmocap == 0
    assert model.opt.density == 0 and model.opt.viscosity == 0 and model.ntendon == 0
    assert not model.opt.disableflags & int(mujoco.mjtDisableBit.mjDSBL_GRAVITY)
    root_joint = int(model.body_jntadr[robot_root])
    assert model.jnt_type[root_joint] == mujoco.mjtJoint.mjJNT_FREE
    position_address = int(model.jnt_qposadr[root_joint])
    velocity_address = int(model.jnt_dofadr[root_joint])
    assert not np.any(model.dof_damping[velocity_address : velocity_address + 6])
    assert not np.any(model.dof_frictionloss[velocity_address : velocity_address + 6])
    robot_bodies = {robot_root}
    for index in range(model.nbody):
        if int(model.body_parentid[index]) in robot_bodies:
            robot_bodies.add(index)
    collision_geometries = [
        index
        for index in range(model.ngeom)
        if int(model.geom_bodyid[index]) in robot_bodies
        and (model.geom_contype[index] or model.geom_conaffinity[index])
    ]
    assert collision_geometries
    window = int(configuration.window)
    assert window >= 5 and window % 2 == 1
    clearance = float(configuration.clearance)
    assert clearance > 0
    assert float(model.geom_margin.max()) < clearance
    mujoco.mj_forward(model, data)
    floor_geometries = [
        index
        for index in range(model.ngeom)
        if int(model.geom_bodyid[index]) not in robot_bodies
        and (model.geom_contype[index] or model.geom_conaffinity[index])
    ]
    assert len(floor_geometries) == 1
    floor = floor_geometries[0]
    assert model.geom_type[floor] == mujoco.mjtGeom.mjGEOM_PLANE
    assert abs(float(data.geom_xpos[floor, 2])) < 1e-8
    np.testing.assert_allclose(
        data.geom_xmat[floor].reshape(3, 3)[:, 2], [0, 0, 1], atol=1e-8
    )
    # Rigidly translate the actual robot along a known ballistic path. This
    # validates COM reconstruction and differentiation without training data.
    calibration_time = np.arange(2 * window + 1) * 0.02
    calibration_time -= calibration_time.mean()
    calibration_centers = []
    for timestamp in calibration_time:
        data.qpos[:] = model.qpos0
        data.qpos[position_address + 2] += (
            1.0 + 0.5 * model.opt.gravity[2] * timestamp**2
        )
        mujoco.mj_forward(model, data)
        calibration_centers.append(data.subtree_com[robot_root, 2])
    calibrated_acceleration = savgol_filter(
        calibration_centers, window, 3, deriv=2, delta=0.02
    )
    calibration_error = float(
        np.abs(calibrated_acceleration - model.opt.gravity[2]).max()
    )
    assert calibration_error < 1e-6
    rows = []
    plots = []
    for split in configuration.splits:
        records = json.loads((root / "motion" / split / "clips.json").read_text())[
            "records"
        ]
        for record in records:
            path = root / "motion" / split / Path(record["motion_path"]).name
            motion = np.load(path)
            assert (
                hashlib.sha256(path.read_bytes()).hexdigest() == record["motion_sha256"]
            )
            timestep = 1 / float(motion["fps"])
            centers = []
            bottoms = []
            for position in motion["qpos"]:
                data.qpos[:] = position
                mujoco.mj_forward(model, data)
                centers.append(data.subtree_com[robot_root].copy())
                low = []
                for index in collision_geometries:
                    kind = model.geom_type[index]
                    size = model.geom_size[index]
                    rotation = data.geom_xmat[index].reshape(3, 3)
                    if kind == mujoco.mjtGeom.mjGEOM_CAPSULE:
                        extent = abs(rotation[2, 2]) * size[1] + size[0]
                    elif kind == mujoco.mjtGeom.mjGEOM_SPHERE:
                        extent = size[0]
                    elif kind == mujoco.mjtGeom.mjGEOM_BOX:
                        extent = np.abs(rotation[2]).dot(size)
                    else:
                        # Bounding sphere is conservative: it may omit airborne
                        # samples but cannot overstate ground clearance.
                        extent = model.geom_rbound[index]
                    low.append(data.geom_xpos[index, 2] - extent)
                bottoms.append(min(low))
            centers = np.asarray(centers)
            airborne = np.asarray(bottoms) > clearance
            sustained = (
                np.convolve(
                    airborne.astype(int), np.ones(window, dtype=int), mode="same"
                )
                == window
            )
            acceleration = savgol_filter(
                centers[:, 2], window, 3, deriv=2, delta=timestep
            )
            mismatch = acceleration[sustained] - float(model.opt.gravity[2])
            if (
                split == "val"
                and record["seed"] == 42001
                and record["task"] in configuration.plot_tasks
            ):
                plots.append(
                    (
                        record["task"],
                        timestep,
                        np.asarray(bottoms),
                        acceleration,
                        sustained,
                    )
                )
            rows.append(
                {
                    "task": record["task"],
                    "seed": record["seed"],
                    "split": split,
                    "motion_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                    "frames": len(centers),
                    "airborne_frames": int(airborne.sum()),
                    "sustained_airborne_frames": int(sustained.sum()),
                    "median_airborne_com_acceleration_z": (
                        float(np.median(acceleration[sustained]))
                        if sustained.any()
                        else None
                    ),
                    "median_gravity_mismatch_m_s2": (
                        float(np.median(np.abs(mismatch))) if sustained.any() else None
                    ),
                    "rms_gravity_mismatch_m_s2": (
                        float(np.sqrt(np.mean(mismatch**2)))
                        if sustained.any()
                        else None
                    ),
                }
            )
    report = {
        "scene_sha256": hashlib.sha256(scene.read_bytes()).hexdigest(),
        "gravity": model.opt.gravity.tolist(),
        "ballistic_calibration_max_error_m_s2": calibration_error,
        "clearance_m": clearance,
        "smoothing": f"Savitzky-Golay window{window}, polynomial3, second derivative; retain only windows whose every sample clears all robot collision geometry by {clearance} m",
        "splits": list(configuration.splits),
        "scope": "Original references, no filtering or corrections; no policy metrics or teacher data",
        "limitations": "Diagnostic of kinematic airborne inconsistency, not a complete feasibility test. Differentiation and retargeting noise contribute. Approximate tracking can remain possible.",
        "episodes": rows,
    }
    (output / "diagnostic.json").write_text(json.dumps(report, indent=2))
    if plots:
        figure, axes = plt.subplots(
            len(plots),
            2,
            figsize=(11, 2.5 * len(plots)),
            squeeze=False,
            constrained_layout=True,
        )
        for row_index, (task, timestep, bottoms, acceleration, sustained) in enumerate(
            plots
        ):
            times = np.arange(len(bottoms)) * timestep
            axes[row_index, 0].plot(times, bottoms)
            axes[row_index, 0].axhline(
                clearance, color="black", linestyle="--", linewidth=1
            )
            axes[row_index, 0].set(
                title=task, ylabel="Lowest collision bound (m)", xlabel="Time (s)"
            )
            axes[row_index, 1].plot(
                times, acceleration, color="0.7", label="Reference COM acceleration"
            )
            axes[row_index, 1].scatter(
                times[sustained],
                acceleration[sustained],
                s=8,
                color="tab:red",
                label="Sustained airborne window",
            )
            axes[row_index, 1].axhline(
                model.opt.gravity[2],
                color="black",
                linestyle="--",
                linewidth=1,
                label="Gravity",
            )
            axes[row_index, 1].set(
                ylabel="Vertical acceleration (m/s²)", xlabel="Time (s)"
            )
            for axis in axes[row_index]:
                axis.grid(alpha=0.2)
        axes[0, 1].legend(fontsize=8)
        figure.suptitle(
            "Unmodified kinematic references; validation seed 42001\nAirborne inconsistency diagnostic, not a complete feasibility classification"
        )
        figure.savefig(output / "airborne_diagnostic.png", dpi=150)
        plt.close(figure)
    for task in sorted({row["task"] for row in rows}):
        selected = [
            row
            for row in rows
            if row["task"] == task and row["sustained_airborne_frames"]
        ]
        if selected:
            print(
                task,
                "clips",
                len(selected),
                "sustained_frames",
                sum(row["sustained_airborne_frames"] for row in selected),
                "median_abs_mismatch",
                float(
                    np.median([row["median_gravity_mismatch_m_s2"] for row in selected])
                ),
                flush=True,
            )


if __name__ == "__main__":
    main()
