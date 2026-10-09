"""Export a normalized CPU policy and its exact observation/action contract."""

import dataclasses
import hashlib
import json
from pathlib import Path
import sys

import hydra
import mujoco
import numpy as np
from omegaconf import DictConfig
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mjlab.rl import MjlabOnPolicyRunner
from mjlab.rl import RslRlVecEnvWrapper
from shared_motion.rl.environment import build_configuration
from shared_motion.rl.environment import TrackingEnvironment
from shared_motion.rl.residual import ReferenceResidualPolicy
from shared_motion.rl.residual import ReferenceResidualWrapper


@hydra.main(
    config_path="../config/tracker_rl", config_name="evaluate", version_base="1.3"
)
def main(configuration: DictConfig):
    torch.set_num_threads(2)
    configuration.num_envs = 1
    output = Path(configuration.output)
    output.mkdir(parents=True, exist_ok=False)
    environment_configuration, agent_configuration = build_configuration(
        configuration, configuration.split, evaluation=True
    )
    environment = TrackingEnvironment(
        cfg=environment_configuration, device=configuration.device
    )
    use_residual = configuration.get("reference_residual", False)
    wrapper_class = ReferenceResidualWrapper if use_residual else RslRlVecEnvWrapper
    wrapper = wrapper_class(environment, clip_actions=agent_configuration.clip_actions)
    runner = MjlabOnPolicyRunner(
        wrapper, dataclasses.asdict(agent_configuration), device=configuration.device
    )
    runner.load(configuration.checkpoint, map_location=configuration.device)
    policy = runner.get_inference_policy(device=configuration.device)
    observations, _ = wrapper.reset()
    with torch.inference_mode():
        predicted = policy(observations)
        expected = (
            wrapper.to_environment_actions(predicted) if use_residual else predicted
        ).cpu()
    action = environment.action_manager.get_term("joint_pos")
    if use_residual:
        runner.export_policy_to_jit(str(output), "policy_core.pt")
        core = torch.jit.load(str(output / "policy_core.pt")).eval()
        combined = ReferenceResidualPolicy(
            core, action.scale[0].cpu(), action.offset[0].cpu()
        ).eval()
        torch.jit.save(torch.jit.script(combined), str(output / "policy.pt"))
    else:
        runner.export_policy_to_jit(str(output), "policy.pt")
    exported = torch.jit.load(str(output / "policy.pt")).eval()
    with torch.inference_mode():
        actual = exported(observations["actor"].cpu())
    torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
    robot = environment.scene["robot"]
    action = environment.action_manager.get_term("joint_pos")
    command = environment.command_manager.get_term("motion")

    def vector(value):
        return (
            value[0].cpu().tolist() if torch.is_tensor(value) else [float(value)] * 29
        )

    contract = {
        "schema_version": 2,
        "reference_convention": "50Hz; body origins in world coordinates; unit wxyz quaternions; joint/body ordering below",
        "reference_residual": bool(use_residual),
        "joint_names": list(robot.joint_names),
        "body_names": list(robot.body_names),
        "tracked_body_names": list(command.cfg.body_names),
        "anchor_body_name": command.cfg.anchor_body_name,
        "joint_qpos_addresses": robot.indexing.joint_q_adr.tolist(),
        "joint_velocity_addresses": robot.indexing.joint_v_adr.tolist(),
        "body_ids": robot.indexing.body_ids.tolist(),
        "default_joint_pos": robot.data.default_joint_pos[0].cpu().tolist(),
        "soft_joint_limits": robot.data.soft_joint_pos_limits[0].cpu().tolist(),
        "action_scale": vector(action.scale),
        "action_offset": vector(action.offset),
        "action_target_names": list(action.target_names),
        "control_timestep": environment.step_dt,
        "preview_offsets": list(configuration.preview_offsets),
        "linear_velocity_sensor": "robot/imu_lin_vel",
        "angular_velocity_sensor": "robot/imu_ang_vel",
        "input_dimensions": observations["actor"].shape[-1],
        "output_dimensions": 29,
        "checkpoint_sha256": hashlib.sha256(
            Path(configuration.checkpoint).read_bytes()
        ).hexdigest(),
        "jit_parity_max_abs": float((actual - expected).abs().max()),
    }
    mujoco.mj_saveModel(environment.sim.mj_model, str(output / "scene.mjb"))
    for name, filename in [("policy", "policy.pt"), ("scene", "scene.mjb")]:
        contract[name + "_sha256"] = hashlib.sha256(
            (output / filename).read_bytes()
        ).hexdigest()
    (output / "contract.json").write_text(json.dumps(contract, indent=2))
    np.savez_compressed(
        output / "parity_fixture.npz",
        observation=observations["actor"].cpu().numpy(),
        action=expected.numpy(),
    )
    print("EXPORT_COMPLETE", contract["jit_parity_max_abs"], flush=True)
    environment.close()


if __name__ == "__main__":
    main()
