"""Real GPU/native input, executed-action history, and partial-reset checks."""

import json
from pathlib import Path
import sys

import hydra
import mujoco
import numpy as np
from omegaconf import DictConfig
from omegaconf import OmegaConf
import torch

repository = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repository))
from shared_motion.rl.cpu import NativeTracker
from shared_motion.rl.environment import build_configuration
from shared_motion.rl.environment import TrackingEnvironment
from shared_motion.rl.sonic_residual import SonicResidualWrapper


@hydra.main(
    config_path="../config/tracker_rl", config_name="sonic_audit", version_base="1.3"
)
def main(arguments: DictConfig):
    if Path(arguments.output).exists():
        raise FileExistsError(arguments.output)
    torch.set_num_threads(2)
    torch.backends.cuda.matmul.allow_tf32 = False
    configuration = OmegaConf.create(
        dict(
            artifacts=arguments.artifacts,
            seed=arguments.seed,
            num_envs=40,
            save_interval=500,
            start_probability=0.35,
            preview_offsets=[5, 10, 20],
        )
    )
    environment_configuration, _ = build_configuration(
        configuration, "val", evaluation=True
    )
    environment = TrackingEnvironment(
        cfg=environment_configuration, device=arguments.device
    )
    wrapper = SonicResidualWrapper(
        environment,
        arguments.sonic_directory,
        Path(arguments.sonic_contract),
    )
    observations, _ = wrapper.reset()
    stored_history = wrapper.histories.clone()
    wrapper.get_observations()
    wrapper.get_observations()
    torch.testing.assert_close(stored_history, wrapper.histories, atol=0, rtol=0)
    evaluation = json.loads(Path(arguments.baseline_metrics).read_text())
    native_configuration = evaluation["configuration"]
    native_configuration["legacy_contract"] = str(Path(arguments.sonic_contract))
    tracker = NativeTracker(native_configuration)
    records = json.loads(
        (Path(arguments.artifacts) / "motion/val/clips.json").read_text()
    )["records"]
    references = [
        dict(
            np.load(
                Path(arguments.artifacts)
                / "motion/val"
                / Path(record["motion_path"]).name
            )
        )
        for record in records
    ]
    command = wrapper.command
    robot = wrapper.robot
    results = []
    for step in range(31):
        if step in [0, 5, 15, 30]:
            maximum_encoder = 0.0
            maximum_state = 0.0
            maximum_actor = 0.0
            for index in range(40):
                mujoco.mj_resetData(tracker.model, tracker.data)
                tracker.data.qpos[:3] = (
                    (
                        robot.data.root_link_pos_w[index]
                        - environment.scene.env_origins[index]
                    )
                    .cpu()
                    .numpy()
                )
                tracker.data.qpos[3:7] = (
                    robot.data.root_link_quat_w[index].cpu().numpy()
                )
                tracker.data.qpos[tracker.joint_addresses] = (
                    robot.data.joint_pos[index].cpu().numpy()
                )
                tracker.data.qvel[:3] = (
                    robot.data.root_link_lin_vel_w[index].cpu().numpy()
                )
                tracker.data.qvel[3:6] = (
                    robot.data.root_link_ang_vel_b[index].cpu().numpy()
                )
                tracker.data.qvel[tracker.velocity_addresses] = (
                    robot.data.joint_vel[index].cpu().numpy()
                )
                mujoco.mj_forward(tracker.model, tracker.data)
                reference = references[int(command.clip_ids[index])]
                frame = int(
                    command.time_steps[index] - command.starts[command.clip_ids[index]]
                )
                native = tracker.observe(
                    reference, frame, wrapper.previous_actions[index].cpu().numpy()
                )
                maximum_encoder = max(
                    maximum_encoder,
                    float(
                        np.abs(
                            native[:1762] - wrapper.encoder_inputs[index].cpu().numpy()
                        ).max()
                    ),
                )
                maximum_state = max(
                    maximum_state,
                    float(
                        np.abs(
                            native[2257:] - wrapper.sonic_states[index].cpu().numpy()
                        ).max()
                    ),
                )
                maximum_actor = max(
                    maximum_actor,
                    float(
                        np.abs(
                            native[1762:2123]
                            - observations["actor"][index, :361].cpu().numpy()
                        ).max()
                    ),
                )
            results.append(
                dict(
                    step=step,
                    encoder=maximum_encoder,
                    state=maximum_state,
                    actor=maximum_actor,
                )
            )
            print(results[-1], flush=True)
            assert max(maximum_encoder, maximum_state, maximum_actor) < 2e-5
        residual = torch.full((40, 29), 0.03, device=arguments.device)
        executed = wrapper.to_environment_actions(residual).clone()
        observations, _, dones, _ = wrapper.step(residual)
        expected_raw = (
            (
                (executed * wrapper.scale + wrapper.offset)[
                    :, wrapper.inverse_action_order
                ]
                - wrapper.base.q0
            )
            / wrapper.base.sonic_scale
        )[:, wrapper.mujoco_to_isaac]
        torch.testing.assert_close(
            wrapper.histories[~dones.bool(), -1, 61:90],
            expected_raw[~dones.bool()],
            atol=1e-6,
            rtol=1e-6,
        )
    command.time_steps[:10] = command.ends[command.clip_ids[:10]] - 1
    previous_histories = wrapper.histories.clone()
    _, _, dones, _ = wrapper.step(torch.zeros(40, 29, device=arguments.device))
    assert bool(dones[:10].all())
    assert bool((~dones.bool()).any())
    torch.testing.assert_close(
        wrapper.histories[~dones.bool(), :-1],
        previous_histories[~dones.bool(), 1:],
        atol=0,
        rtol=0,
    )
    histories = wrapper.histories[dones.bool()]
    torch.testing.assert_close(
        histories, histories[:, -1:].expand_as(histories), atol=0, rtol=0
    )
    assert bool((histories[:, :, 61:90] == 0).all())
    report = dict(
        passed=True,
        results=results,
        forced_resets=int(dones[:10].sum()),
        checks=[
            "idempotent reads",
            "native current/preview and SONIC inputs",
            "actual residual-modified action histories",
            "partial environment reset",
        ],
    )
    output = Path(arguments.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print("SONIC_WRAPPER_AUDIT_PASS", flush=True)
    environment.close()


if __name__ == "__main__":
    main()
