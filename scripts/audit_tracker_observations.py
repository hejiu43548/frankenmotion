"""Audit native observation reconstruction across actual GPU trajectories."""

import hydra
from omegaconf import DictConfig
import json
from pathlib import Path
import sys
import mujoco
import numpy as np
from omegaconf import OmegaConf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared_motion.rl.cpu import NativeTracker, rotation
from shared_motion.rl.environment import build_configuration, TrackingEnvironment


@hydra.main(
    config_path="../config/tracker_rl",
    config_name="observation_audit",
    version_base="1.3",
)
def main(arguments: DictConfig):
    artifacts = Path(arguments.artifacts)
    export = Path(arguments.export)
    contract = json.loads((export / "contract.json").read_text())
    records = json.loads(
        (artifacts / "motion" / arguments.split / "clips.json").read_text()
    )["records"]
    output = Path(arguments.output)
    if output.exists():
        raise FileExistsError(output)
    torch.set_num_threads(2)
    configuration = OmegaConf.create(
        dict(
            artifacts=str(artifacts),
            seed=arguments.seed,
            num_envs=len(records),
            save_interval=500,
            start_probability=0.35,
            preview_offsets=contract["preview_offsets"],
        )
    )
    environment_configuration, _ = build_configuration(
        configuration, arguments.split, evaluation=True
    )
    environment = TrackingEnvironment(
        cfg=environment_configuration, device=arguments.device
    )
    policy = torch.jit.load(
        str(export / "policy.pt"), map_location=arguments.device
    ).eval()
    tracker = NativeTracker(
        dict(
            policy=str(export / "policy.pt"),
            contract=str(export / "contract.json"),
            scene=str(export / "scene.mjb"),
            policy_kind="rl",
            seed=61001,
            perturbation=0,
        )
    )
    references = [
        dict(
            np.load(
                artifacts
                / "motion"
                / arguments.split
                / Path(record["motion_path"]).name
            )
        )
        for record in records
    ]
    observations, _ = environment.reset()
    command = environment.command_manager.get_term("motion")
    robot = environment.scene["robot"]
    results = []
    for step in range(max(arguments.steps) + 1):
        if step in arguments.steps:
            position = (
                (robot.data.root_link_pos_w - environment.scene.env_origins)
                .cpu()
                .numpy()
            )
            quaternion = robot.data.root_link_quat_w.cpu().numpy()
            linear_velocity = robot.data.root_link_lin_vel_w.cpu().numpy()
            angular_velocity = robot.data.root_link_ang_vel_w.cpu().numpy()
            joint_positions = robot.data.joint_pos.cpu().numpy()
            joint_velocities = robot.data.joint_vel.cpu().numpy()
            previous_actions = environment.action_manager.action.cpu().numpy()
            expected = observations["actor"].cpu().numpy()
            differences = []
            for index in range(len(records)):
                reference = references[int(command.clip_ids[index])]
                frame = int(
                    command.time_steps[index] - command.starts[command.clip_ids[index]]
                )
                mujoco.mj_resetData(tracker.model, tracker.data)
                tracker.data.qpos[:3] = position[index]
                tracker.data.qpos[3:7] = quaternion[index]
                tracker.data.qpos[tracker.joint_addresses] = joint_positions[index]
                tracker.data.qvel[:3] = linear_velocity[index]
                tracker.data.qvel[3:6] = (
                    rotation(quaternion[index]).inv().apply(angular_velocity[index])
                )
                tracker.data.qvel[tracker.velocity_addresses] = joint_velocities[index]
                mujoco.mj_forward(tracker.model, tracker.data)
                actual = tracker.observe(reference, frame, previous_actions[index])
                differences.append(np.abs(actual - expected[index]))
            differences = np.asarray(differences)
            results.append(
                dict(
                    step=step,
                    maximum=float(differences.max()),
                    current_maximum=float(differences[:, :160].max()),
                    preview_maximum=(
                        float(differences[:, 160:].max())
                        if differences.shape[1] > 160
                        else 0.0
                    ),
                    worst_dimension=int(
                        np.unravel_index(differences.argmax(), differences.shape)[1]
                    ),
                )
            )
        if step < max(arguments.steps):
            with torch.inference_mode():
                actions = policy(observations["actor"])
            observations, _, _, _, _ = environment.step(actions)
    output.parent.mkdir(parents=True, exist_ok=True)
    passed = max(row["maximum"] for row in results) <= arguments.tolerance
    output.write_text(
        json.dumps(
            {
                "passed": passed,
                "environments": len(records),
                "tolerance": arguments.tolerance,
                "results": results,
            },
            indent=2,
        )
    )
    print(json.dumps(results, indent=2))
    environment.close()

    if not passed:
        raise RuntimeError("Native and GPU observation construction disagree")


if __name__ == "__main__":
    main()
