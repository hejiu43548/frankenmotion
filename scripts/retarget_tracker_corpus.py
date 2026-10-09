"""Uniform GMR conversion of official references, without teacher-policy imports."""

import argparse
from concurrent.futures import ProcessPoolExecutor
import contextlib
import hashlib
import io
import json
from pathlib import Path
import sys
import types

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation

BODY_NAMES = (
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
)
PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)


def convert_record(arguments):
    record, configuration = arguments
    # The upstream package __init__ eagerly imports optional viewers. Load its
    # computational submodule without these unrelated rendering dependencies.
    package = types.ModuleType("general_motion_retargeting")
    package.__path__ = [str(Path(configuration["gmr"]) / "general_motion_retargeting")]
    sys.modules[package.__name__] = package
    sys.path.insert(0, configuration["dependencies"])
    from general_motion_retargeting.motion_retarget import GeneralMotionRetargeting

    source = np.load(record["path"])
    joints = source["joints"].astype(np.float64)
    rotations = (
        Rotation.from_rotvec(source["poses"].reshape(-1, 3))
        .as_matrix()
        .reshape(-1, 22, 3, 3)
    )
    lateral = joints[0, 1] - joints[0, 2]
    heading = np.arctan2(lateral[1], lateral[0]) - np.pi / 2
    alignment = Rotation.from_euler("z", -heading).as_matrix()
    joints = np.einsum("ij,tkj->tki", alignment, joints)
    joints[:, :, :2] -= joints[0, 0, :2].copy()
    rotations[:, 0] = alignment @ rotations[:, 0]
    world_rotations = []
    for index, parent in enumerate(PARENTS):
        world_rotations.append(
            rotations[:, index]
            if parent < 0
            else world_rotations[parent] @ rotations[:, index]
        )
    quaternions = (
        Rotation.from_matrix(np.stack(world_rotations, axis=1).reshape(-1, 3, 3))
        .as_quat()[:, [3, 0, 1, 2]]
        .reshape(-1, 22, 4)
    )
    with contextlib.redirect_stdout(io.StringIO()):
        retargeter = GeneralMotionRetargeting("smplx", "unitree_g1", verbose=False)
    neutral = mujoco.MjData(retargeter.model)
    mujoco.mj_forward(retargeter.model, neutral)
    shoulders = [
        retargeter.model.body(name + "_shoulder_roll_link").id
        for name in ["left", "right"]
    ]
    ankles = [
        retargeter.model.body(name + "_ankle_roll_link").id
        for name in ["left", "right"]
    ]
    robot_height = neutral.xpos[shoulders, 2].mean() - neutral.xpos[ankles, 2].mean()
    human_height = float(np.load(configuration["skeleton"])["height"])
    scale = robot_height / human_height
    retargeter.human_scale_table = {
        name: scale for name in retargeter.human_scale_table
    }
    states = []
    for frame_index in range(len(joints)):
        frame = {
            name: (
                joints[frame_index, joint_index],
                quaternions[frame_index, joint_index],
            )
            for joint_index, name in enumerate(BODY_NAMES)
        }
        states.append(retargeter.retarget(frame, offset_to_ground=False))
    states = np.asarray(states)
    if not np.isfinite(states).all():
        raise ValueError(f'Nonfinite retargeting: {record["path"]}')
    joint_names = [
        retargeter.model.joint(index).name for index in range(1, retargeter.model.njnt)
    ]
    destination = Path(configuration["output"]) / (Path(record["path"]).stem + ".npz")
    np.savez_compressed(destination, qpos=states, joint_names=joint_names, fps=20)
    return dict(
        record,
        reference_path=str(destination),
        uniform_scale=float(scale),
        reference_sha256=hashlib.sha256(destination.read_bytes()).hexdigest(),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--gmr", required=True)
    parser.add_argument("--dependencies", required=True)
    parser.add_argument("--skeleton", required=True)
    parser.add_argument("--workers", type=int, default=4)
    configuration = vars(parser.parse_args())
    output = Path(configuration["output"])
    output.mkdir(parents=True, exist_ok=False)
    records = json.loads(Path(configuration["manifest"]).read_text())
    converted = []
    with ProcessPoolExecutor(configuration["workers"]) as pool:
        for record in pool.map(
            convert_record, [(record, configuration) for record in records]
        ):
            converted.append(record)
            (output / "manifest.json").write_text(json.dumps(converted, indent=2))
            print(f"RETARGETED {len(converted)}/{len(records)}", flush=True)
    (output / "complete.json").write_text(json.dumps({"count": len(converted)}))


if __name__ == "__main__":
    main()
