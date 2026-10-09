"""Resample GMR references and compute FK in the exact training robot model."""

import hydra
from omegaconf import DictConfig
import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from scipy.spatial.transform import Slerp

KEYS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
)


def convert(reference, model, contract):
    data = mujoco.MjData(model)
    source = np.load(reference)
    states = source["qpos"]
    times = np.arange(len(states)) / float(source["fps"])
    target_times = np.arange(0, times[-1] + 1e-9, 0.02)
    positions = np.stack(
        [np.interp(target_times, times, column) for column in states[:, :3].T], axis=1
    )
    rotations = Slerp(times, Rotation.from_quat(states[:, [4, 5, 6, 3]]))(target_times)
    quaternions = rotations.as_quat()[:, [3, 0, 1, 2]]
    source_names = source["joint_names"].tolist()
    order = [source_names.index(name) for name in contract["joint_names"]]
    joint_positions = np.stack(
        [np.interp(target_times, times, column) for column in states[:, 7:].T], axis=1
    )[:, order]
    qpos = np.tile(model.qpos0, (len(target_times), 1))
    qpos[:, :3] = positions
    qpos[:, 3:7] = quaternions
    qpos[:, contract["joint_qpos_addresses"]] = joint_positions
    data.qpos[:] = qpos[0]
    mujoco.mj_forward(model, data)
    foot_minima = []
    for geom_index in range(model.ngeom):
        body_name = model.body(int(model.geom_bodyid[geom_index])).name
        if "ankle_roll" not in body_name or not (
            model.geom_contype[geom_index] or model.geom_conaffinity[geom_index]
        ):
            continue
        position = data.geom_xpos[geom_index]
        orientation = data.geom_xmat[geom_index].reshape(3, 3)
        size = model.geom_size[geom_index]
        if model.geom_type[geom_index] == mujoco.mjtGeom.mjGEOM_SPHERE:
            foot_minima.append(position[2] - size[0])
        elif model.geom_type[geom_index] == mujoco.mjtGeom.mjGEOM_CAPSULE:
            foot_minima.append(position[2] - abs(orientation[2, 2]) * size[1] - size[0])
        elif model.geom_type[geom_index] == mujoco.mjtGeom.mjGEOM_BOX:
            foot_minima.append(position[2] - np.abs(orientation[2]).dot(size))
    if not foot_minima:
        raise ValueError("No supported foot collision geometries")
    qpos[:, 2] -= min(foot_minima)
    velocities = np.zeros((len(qpos), model.nv))
    for frame_index in range(len(qpos) - 1):
        mujoco.mj_differentiatePos(
            model,
            velocities[frame_index],
            0.02,
            qpos[frame_index],
            qpos[frame_index + 1],
        )
    velocities[-1] = velocities[-2]
    arrays = {key: [] for key in KEYS}
    body_velocity = np.empty(6)
    for frame_index in range(len(qpos)):
        data.qpos[:] = qpos[frame_index]
        data.qvel[:] = velocities[frame_index]
        mujoco.mj_forward(model, data)
        linear = []
        angular = []
        for body_id in contract["body_ids"]:
            mujoco.mj_objectVelocity(
                model, data, mujoco.mjtObj.mjOBJ_XBODY, body_id, body_velocity, 0
            )
            angular.append(body_velocity[:3].copy())
            linear.append(body_velocity[3:].copy())
        values = [
            joint_positions[frame_index],
            velocities[frame_index, contract["joint_velocity_addresses"]],
            data.xpos[contract["body_ids"]].copy(),
            data.xquat[contract["body_ids"]].copy(),
            linear,
            angular,
        ]
        for key, value in zip(KEYS, values):
            arrays[key].append(value)
    arrays = {key: np.asarray(value, dtype=np.float32) for key, value in arrays.items()}
    if not all(np.isfinite(value).all() for value in arrays.values()):
        raise ValueError("Invalid FK data")
    return arrays, qpos


@hydra.main(
    config_path="../config/tracker_rl", config_name="prepare_motion", version_base="1.3"
)
def main(configuration: DictConfig):
    root = Path(configuration.root)
    contract = json.loads((root / "scene/contract.json").read_text())
    model = mujoco.MjModel.from_binary_path(str(root / "scene/scene.mjb"))
    manifest = json.loads((root / "retarget/manifest.json").read_text())
    output = root / "motion"
    output.mkdir(exist_ok=False)
    for split in ["train", "val", "test"]:
        directory = output / split
        directory.mkdir()
        chunks = {key: [] for key in KEYS}
        records = []
        ends = []
        for record in [record for record in manifest if record["split"] == split]:
            reference_path = root / "retarget" / Path(record["reference_path"]).name
            if (
                hashlib.sha256(reference_path.read_bytes()).hexdigest()
                != record["reference_sha256"]
            ):
                raise ValueError(f"Retarget checksum mismatch: {reference_path}")
            arrays, states = convert(reference_path, model, contract)
            destination = directory / (Path(record["path"]).stem + ".npz")
            np.savez_compressed(destination, fps=50, qpos=states, **arrays)
            records.append(
                dict(
                    record,
                    motion_path=str(destination),
                    motion_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
                )
            )
            for key in KEYS:
                chunks[key].append(arrays[key])
            ends.append((ends[-1] if ends else 0) + len(states))
        np.savez_compressed(
            directory / "motions.npz",
            fps=50,
            **{key: np.concatenate(value) for key, value in chunks.items()},
        )
        (directory / "clips.json").write_text(
            json.dumps({"ends": ends, "records": records}, indent=2)
        )
        print("PREPARED", split, len(records), ends[-1], flush=True)
    (output / "complete.json").write_text(
        json.dumps(
            {
                "count": len(manifest),
                "scene_sha256": hashlib.sha256(
                    (root / "scene/scene.mjb").read_bytes()
                ).hexdigest(),
            }
        )
    )


if __name__ == "__main__":
    main()
