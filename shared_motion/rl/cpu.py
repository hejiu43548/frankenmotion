"""Native MuJoCo inference for RL policies and the archived shared20 baseline.

The archived policy is evaluation-only. No actions or trajectories from it are
accepted by the training data interface.
"""

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
import torch


def rotation(quaternion):
    return Rotation.from_quat(np.asarray(quaternion)[[1, 2, 3, 0]])


class NativeTracker:
    def __init__(self, configuration):
        torch.set_num_threads(1)
        self.configuration = configuration
        self.contract = json.loads(Path(configuration["contract"]).read_text())
        self.model = mujoco.MjModel.from_binary_path(configuration["scene"])
        self.data = mujoco.MjData(self.model)
        self.policy = torch.jit.load(configuration["policy"], map_location="cpu").eval()
        self.legacy = configuration["policy_kind"] == "shared20"
        self.joint_addresses = self.contract["joint_qpos_addresses"]
        self.velocity_addresses = self.contract["joint_velocity_addresses"]
        self.anchor = self.model.body("robot/" + self.contract["anchor_body_name"]).id
        self.anchor_reference = self.contract["body_names"].index(
            self.contract["anchor_body_name"]
        )
        self.tracked_reference = [
            self.contract["body_names"].index(name)
            for name in self.contract["tracked_body_names"]
        ]
        self.tracked_bodies = [
            self.model.body("robot/" + name).id
            for name in self.contract["tracked_body_names"]
        ]
        self.end_effectors = [
            self.model.body("robot/" + name).id
            for name in [
                "left_ankle_roll_link",
                "right_ankle_roll_link",
                "left_wrist_yaw_link",
                "right_wrist_yaw_link",
            ]
        ]
        self.end_effector_references = [
            self.contract["body_names"].index(
                self.model.body(index).name.removeprefix("robot/")
            )
            for index in self.end_effectors
        ]
        self.actuators = [
            int(
                np.flatnonzero(
                    self.model.actuator_trnid[:, 0]
                    == self.model.joint("robot/" + name).id
                )[0]
            )
            for name in self.contract["action_target_names"]
        ]
        if self.legacy:
            self.legacy_contract = json.loads(
                Path(configuration["legacy_contract"]).read_text()
            )
            for name in ["joint_names", "action_target_names"]:
                if self.contract[name] != self.legacy_contract[name]:
                    raise ValueError(f"Legacy action ABI differs: {name}")
            for name in ["action_scale", "action_offset", "default_joint_pos"]:
                np.testing.assert_allclose(
                    self.contract[name], self.legacy_contract[name], atol=1e-7
                )
            self.preview_offsets = json.loads(
                Path(configuration["policy"]).with_suffix(".json").read_text()
            )["preview_offsets"]
        else:
            self.preview_offsets = self.contract["preview_offsets"]

    def sensor(self, name):
        sensor = self.model.sensor(name)
        return self.data.sensordata[
            sensor.adr[0] : sensor.adr[0] + sensor.dim[0]
        ].copy()

    def observe(self, reference, frame_index, previous_action):
        anchor_inverse = rotation(self.data.xquat[self.anchor]).inv()

        def relative(index):
            position = anchor_inverse.apply(
                reference["body_pos_w"][index, self.anchor_reference]
                - self.data.xpos[self.anchor]
            )
            orientation = (
                (
                    anchor_inverse
                    * rotation(reference["body_quat_w"][index, self.anchor_reference])
                )
                .as_matrix()[:, :2]
                .ravel()
            )
            return [position, orientation]

        values = [
            reference["joint_pos"][frame_index],
            reference["joint_vel"][frame_index],
            *relative(frame_index),
            self.sensor(self.contract["linear_velocity_sensor"]),
            self.sensor(self.contract["angular_velocity_sensor"]),
            self.data.qpos[self.joint_addresses] - self.contract["default_joint_pos"],
            self.data.qvel[self.velocity_addresses],
            previous_action,
        ]
        for offset in self.preview_offsets:
            index = min(frame_index + offset, len(reference["joint_pos"]) - 1)
            values.extend(
                [
                    reference["joint_pos"][index],
                    reference["joint_vel"][index],
                    *relative(index),
                ]
            )
        observation = np.concatenate(values)
        if not self.legacy:
            return observation.astype(np.float32)
        contract = self.legacy_contract
        source_order = [
            contract["joint_names"].index(name)
            for name in contract["sonic_source_joint_names"]
        ]
        action_order = [
            contract["sonic_source_joint_names"].index(name)
            for name in contract["action_target_names"]
        ]
        mujoco_to_isaac = np.argsort(self.policy.il_to_mj.numpy())
        indices = np.minimum(
            frame_index + np.arange(10) * 5, len(reference["joint_pos"]) - 1
        )
        encoder = np.zeros(1762, dtype=np.float32)
        encoder[4:294] = reference["joint_pos"][indices][:, source_order][
            :, mujoco_to_isaac
        ].ravel()
        encoder[294:584] = reference["joint_vel"][indices][:, source_order][
            :, mujoco_to_isaac
        ].ravel()
        root_rotation = rotation(self.data.qpos[3:7])
        future_rotations = Rotation.from_quat(
            reference["body_quat_w"][indices, 0][:, [1, 2, 3, 0]]
        )
        encoder[601:661] = (
            (root_rotation.inv() * future_rotations).as_matrix()[:, :, :2].ravel()
        )
        if frame_index == 0:
            previous_sonic = np.zeros(29)
        else:
            targets = (
                previous_action * np.asarray(contract["action_scale"])
                + contract["action_offset"]
            )
            previous_sonic = (
                (
                    targets[np.argsort(action_order)]
                    - contract["sonic_default_positions"]
                )
                / np.asarray(contract["sonic_action_scale"])
            )[mujoco_to_isaac]
        state = np.concatenate(
            [
                self.data.qvel[3:6],
                (
                    self.data.qpos[self.joint_addresses][source_order]
                    - contract["sonic_default_positions"]
                )[mujoco_to_isaac],
                self.data.qvel[self.velocity_addresses][source_order][mujoco_to_isaac],
                previous_sonic,
                root_rotation.inv().apply([0, 0, -1]),
            ]
        )
        return np.concatenate([encoder, observation, state]).astype(np.float32)

    def reset_reference(self, reference, record):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = reference["qpos"][0]
        limits = np.asarray(self.contract["soft_joint_limits"])
        self.data.qpos[self.joint_addresses] = np.clip(
            self.data.qpos[self.joint_addresses], limits[:, 0], limits[:, 1]
        )
        self.data.qvel[:3] = reference["body_lin_vel_w"][0, 0]
        self.data.qvel[3:6] = (
            rotation(self.data.qpos[3:7]).inv().apply(reference["body_ang_vel_w"][0, 0])
        )
        self.data.qvel[self.velocity_addresses] = reference["joint_vel"][0]
        if self.configuration["perturbation"]:
            random = np.random.default_rng(self.configuration["seed"] + record["seed"])
            self.data.qvel[:6] += random.uniform(
                -self.configuration["perturbation"],
                self.configuration["perturbation"],
                6,
            )
        mujoco.mj_forward(self.model, self.data)
        if self.legacy:
            self.policy.reset()

    def run(self, record):
        motion_path = (
            Path(self.configuration["artifacts"])
            / "motion"
            / record["split"]
            / Path(record["motion_path"]).name
        )
        if (
            hashlib.sha256(motion_path.read_bytes()).hexdigest()
            != record["motion_sha256"]
        ):
            raise ValueError(f"Motion checksum mismatch: {motion_path}")
        reference = dict(np.load(motion_path))
        return self.run_reference(reference, record)

    def run_reference(self, reference, record):
        """Track one prepared reference, also usable outside dataset evaluation."""
        self.reset_reference(reference, record)
        previous_action = np.zeros(29)
        states = []
        actions = []
        errors = []
        failures = []
        horizon = len(reference["joint_pos"])
        for frame_index in range(horizon):
            observation = self.observe(reference, frame_index, previous_action)
            with torch.inference_mode():
                action = self.policy(torch.from_numpy(observation)[None])[0].numpy()
            if not np.isfinite(action).all():
                raise RuntimeError("Nonfinite policy output")
            errors.append(
                [
                    np.linalg.norm(
                        reference["body_pos_w"][frame_index, self.anchor_reference]
                        - self.data.xpos[self.anchor]
                    ),
                    np.linalg.norm(
                        reference["body_pos_w"][frame_index, self.tracked_reference]
                        - self.data.xpos[self.tracked_bodies],
                        axis=-1,
                    ).mean(),
                    np.sqrt(
                        np.mean(
                            (
                                reference["joint_pos"][frame_index]
                                - self.data.qpos[self.joint_addresses]
                            )
                            ** 2
                        )
                    ),
                    np.mean((action - previous_action) ** 2),
                ]
            )
            states.append(self.data.qpos.copy())
            actions.append(action.copy())
            self.data.ctrl[self.actuators] = (
                action * np.asarray(self.contract["action_scale"])
                + self.contract["action_offset"]
            )
            for _ in range(
                round(self.contract["control_timestep"] / self.model.opt.timestep)
            ):
                mujoco.mj_step(self.model, self.data)
            mujoco.mj_forward(self.model, self.data)
            if not np.isfinite(self.data.qpos).all():
                failures.append("nonfinite")
            if (
                abs(
                    reference["body_pos_w"][frame_index, self.anchor_reference, 2]
                    - self.data.xpos[self.anchor, 2]
                )
                > 0.25
            ):
                failures.append("anchor_pos")
            actual_gravity = (
                rotation(self.data.xquat[self.anchor]).inv().apply([0, 0, -1])
            )
            target_gravity = (
                rotation(reference["body_quat_w"][frame_index, self.anchor_reference])
                .inv()
                .apply([0, 0, -1])
            )
            if abs(actual_gravity[2] - target_gravity[2]) > 0.8:
                failures.append("anchor_ori")
            if (
                np.max(
                    np.abs(
                        reference["body_pos_w"][
                            frame_index, self.end_effector_references, 2
                        ]
                        - self.data.xpos[self.end_effectors, 2]
                    )
                )
                > 0.25
            ):
                failures.append("ee_body_pos")
            if failures:
                break
            previous_action = action
        errors = np.asarray(errors)
        result = {
            name: float(value)
            for name, value in zip(
                ["root_m", "body_m", "joint_rad", "action_delta_squared"],
                errors.mean(axis=0),
            )
        }
        for index, (name, penalty) in enumerate(
            [("root_m", 0.5), ("body_m", 0.5), ("joint_rad", 1.0)]
        ):
            result[name + "_failure_penalized"] = float(
                (errors[:, index].sum() + (horizon - len(errors)) * penalty) / horizon
            )
        result.update(
            task=record["task"],
            motion_seed=record["seed"],
            frames=len(states),
            expected_frames=horizon,
            complete=not failures,
            failure=failures,
            tracking_success=not failures
            and result["root_m"] < 0.15
            and result["body_m"] < 0.2
            and result["joint_rad"] < 0.4,
        )
        return result, np.asarray(states), np.asarray(actions)
