"""Run a prepared 50 Hz G1 reference using one exported tracker and fixed dynamics."""

import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
import torch
import mujoco
from scipy.spatial.transform import Rotation
from .runtime import Tracker


def run_exported_tracker(actor, scene, contract, reference, initial_state, output):
    """Use the RL export contract while preserving the supplied simulator state."""
    from shared_motion.rl.cpu import NativeTracker

    metadata = json.loads(Path(contract).read_text())
    policy_kind = metadata.get("policy_kind")
    if policy_kind not in ["rl", "sonic_rl"]:
        raise ValueError("Expected an RL export with policy_kind rl or sonic_rl")
    tracker = NativeTracker(
        {
            "policy": str(actor),
            "policy_kind": policy_kind,
            "contract": str(contract),
            "scene": str(scene),
            "legacy_contract": str(Path(actor).parent / "sonic_contract.json"),
            "seed": 61001,
            "perturbation": 0.0,
        }
    )
    with np.load(reference) as archive:
        reference_motion = dict(archive)
    with np.load(initial_state) as archive:
        supplied_state = dict(archive)
    result, states, actions = tracker.run_reference(
        reference_motion, {"task": "provided_reference", "seed": 0}, supplied_state
    )
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, qpos=states, actions=actions, fps=50.0)
    final_state_path = output.with_name(output.stem + "_final_state.npz")
    np.savez_compressed(
        final_state_path, qpos=tracker.data.qpos, qvel=tracker.data.qvel
    )
    output.with_suffix(".json").write_text(
        json.dumps(
            {
                "complete": result["complete"],
                "termination_time": (
                    float(tracker.data.time) if result["failure"] else None
                ),
                "frames": len(states),
                "tracker": str(actor),
                "reference": str(reference),
                "state_reset_after_initialization": False,
                "initialization": "provided qpos and qvel; missing qvel defaults to zero",
                "frame_recording": "pre-control states; final post-control state saved separately",
                "final_state": str(final_state_path),
                "termination_protocol": "native RL tracking thresholds, including reference deviation",
                "episode": result,
                "input_sha256": {
                    name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                    for name, path in [
                        ("policy", actor),
                        ("scene", scene),
                        ("contract", contract),
                        ("reference", reference),
                        ("initial_state", initial_state),
                    ]
                },
            },
            indent=2,
        )
    )


def run(actor, scene, contract, reference, initial_state, output):
    torch.set_num_threads(1)
    inference_contract = json.loads(Path(contract).read_text())
    if inference_contract.get("schema_version") == 2:
        return run_exported_tracker(
            actor, scene, contract, reference, initial_state, output
        )
    tracker = Tracker(actor)
    actor_model = tracker.actor
    inference_contract["preview_offsets"] = json.loads(
        Path(actor).with_suffix(".json").read_text()
    )["preview_offsets"]
    simulation_model = mujoco.MjModel.from_binary_path(str(scene))
    simulation_data = mujoco.MjData(simulation_model)
    reference_motion = dict(np.load(reference))
    initial_qpos = np.load(initial_state)["qpos"]
    simulation_data.qpos[:] = (
        initial_qpos[0] if initial_qpos.ndim == 2 else initial_qpos
    )
    mujoco.mj_forward(simulation_model, simulation_data)
    joint_position_addresses = [
        simulation_model.joint("robot/" + joint_name).qposadr[0]
        for joint_name in inference_contract["joint_names"]
    ]
    joint_velocity_addresses = [
        simulation_model.joint("robot/" + joint_name).dofadr[0]
        for joint_name in inference_contract["joint_names"]
    ]
    actuator_ids = [
        int(
            np.where(
                simulation_model.actuator_trnid[:, 0]
                == simulation_model.joint("robot/" + joint_name).id
            )[0][0]
        )
        for joint_name in inference_contract["action_target_names"]
    ]
    anchor_body_id = simulation_model.body(inference_contract["anchor_body_name"]).id
    pelvis_body_id = simulation_model.body("robot/pelvis").id
    source_joint_names = inference_contract["sonic_source_joint_names"]
    source_joint_order = [
        inference_contract["joint_names"].index(joint_name)
        for joint_name in source_joint_names
    ]
    action_order = [
        source_joint_names.index(joint_name)
        for joint_name in inference_contract["action_target_names"]
    ]
    mj_to_il = np.argsort(actor_model.il_to_mj.numpy())
    last_action = np.zeros(29)
    num_frames = len(reference_motion["joint_pos"])
    states = [simulation_data.qpos.copy()]
    actions = []
    failure = None

    def rotation_from_wxyz(quaternion):
        return Rotation.from_quat(np.asarray(quaternion)[[1, 2, 3, 0]])

    def sensor(name):
        sensor_spec = simulation_model.sensor(name)
        return simulation_data.sensordata[
            sensor_spec.adr[0] : sensor_spec.adr[0] + sensor_spec.dim[0]
        ].copy()

    for frame_index in range(num_frames - 1):
        inverse_anchor_rotation = rotation_from_wxyz(
            simulation_data.xquat[anchor_body_id]
        ).inv()

        def relative(reference_frame):
            return inverse_anchor_rotation.apply(
                reference_motion["body_pos_w"][
                    reference_frame, inference_contract["reference_anchor_index"]
                ]
                - simulation_data.xpos[anchor_body_id]
            ), (
                inverse_anchor_rotation
                * rotation_from_wxyz(
                    reference_motion["body_quat_w"][
                        reference_frame, inference_contract["reference_anchor_index"]
                    ]
                )
            ).as_matrix()[
                :, :2
            ].reshape(
                -1
            )

        relative_position, relative_rotation = relative(frame_index)
        observation_parts = [
            reference_motion["joint_pos"][frame_index],
            reference_motion["joint_vel"][frame_index],
            relative_position,
            relative_rotation,
            sensor(inference_contract["linear_velocity_sensor"]),
            sensor(inference_contract["angular_velocity_sensor"]),
            simulation_data.qpos[joint_position_addresses]
            - np.array(inference_contract["default_joint_pos"]),
            simulation_data.qvel[joint_velocity_addresses],
            last_action,
        ]
        for offset in inference_contract["preview_offsets"]:
            preview_frame = min(frame_index + offset, num_frames - 1)
            relative_position, relative_rotation = relative(preview_frame)
            observation_parts.extend(
                [
                    reference_motion["joint_pos"][preview_frame],
                    reference_motion["joint_vel"][preview_frame],
                    relative_position,
                    relative_rotation,
                ]
            )
        encoder_frame_indices = np.minimum(
            frame_index + np.arange(10) * 5, num_frames - 1
        )
        encoder_input = np.zeros(1762, np.float32)
        encoder_input[4:294] = reference_motion["joint_pos"][encoder_frame_indices][
            :, source_joint_order
        ][:, mj_to_il].ravel()
        encoder_input[294:584] = reference_motion["joint_vel"][encoder_frame_indices][
            :, source_joint_order
        ][:, mj_to_il].ravel()
        robot_rotation = rotation_from_wxyz(simulation_data.qpos[3:7])
        reference_rotations = Rotation.from_quat(
            reference_motion["body_quat_w"][encoder_frame_indices, 0][:, [1, 2, 3, 0]]
        )
        encoder_input[601:661] = (
            (robot_rotation.inv() * reference_rotations).as_matrix()[:, :, :2].ravel()
        )
        previous_sonic_action = (
            np.zeros(29)
            if frame_index == 0
            else (
                (
                    (
                        last_action * np.array(inference_contract["action_scale"])
                        + np.array(inference_contract["action_offset"])
                    )[np.argsort(action_order)]
                    - np.array(inference_contract["sonic_default_positions"])
                )
                / np.array(inference_contract["sonic_action_scale"])
            )[mj_to_il]
        )
        state = np.r_[
            simulation_data.qvel[3:6],
            (
                simulation_data.qpos[joint_position_addresses][source_joint_order]
                - np.array(inference_contract["sonic_default_positions"])
            )[mj_to_il],
            simulation_data.qvel[joint_velocity_addresses][source_joint_order][
                mj_to_il
            ],
            previous_sonic_action,
            robot_rotation.inv().apply([0, 0, -1]),
        ]
        observation = np.r_[
            encoder_input, np.concatenate(observation_parts), state
        ].astype(np.float32)
        last_action = tracker(observation[None])[0].numpy()
        actions.append(last_action.copy())
        simulation_data.ctrl[actuator_ids] = last_action * np.array(
            inference_contract["action_scale"]
        ) + np.array(inference_contract["action_offset"])
        for _ in range(
            round(
                inference_contract["control_timestep"] / simulation_model.opt.timestep
            )
        ):
            mujoco.mj_step(simulation_model, simulation_data)
        mujoco.mj_forward(simulation_model, simulation_data)
        states.append(simulation_data.qpos.copy())
        tilt = np.arccos(
            np.clip(simulation_data.xmat[pelvis_body_id].reshape(3, 3)[2, 2], -1, 1)
        )
        if (
            not np.isfinite(simulation_data.qpos).all()
            or simulation_data.xpos[pelvis_body_id, 2] < 0.35
            or tilt > np.pi / 3
        ):
            failure = float(simulation_data.time)
            break
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, qpos=states, actions=actions, fps=50.0)
    output.with_suffix(".json").write_text(
        json.dumps(
            dict(
                complete=failure is None,
                termination_time=failure,
                frames=len(states),
                tracker=str(actor),
                reference=str(reference),
                state_reset_after_initialization=False,
            ),
            indent=2,
        )
    )


def main():
    parser = argparse.ArgumentParser()
    for name in ["actor", "scene", "contract", "reference", "initial-state", "output"]:
        parser.add_argument("--" + name, required=True)
    args = parser.parse_args()
    run(
        args.actor,
        args.scene,
        args.contract,
        args.reference,
        args.initial_state,
        args.output,
    )


if __name__ == "__main__":
    main()
