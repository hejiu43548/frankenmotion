"""Full-start evaluation, paired across policies, with failure-inclusive errors."""

import dataclasses
import hashlib
import json
from pathlib import Path
import sys

import hydra
import numpy as np
from omegaconf import DictConfig
from omegaconf import OmegaConf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mjlab.rl import MjlabOnPolicyRunner
from shared_motion.rl.environment import build_configuration
from shared_motion.rl.environment import TrackingEnvironment
from shared_motion.rl.wrappers import make_wrapper


@hydra.main(
    config_path="../config/tracker_rl", config_name="evaluate", version_base="1.3"
)
def main(configuration: DictConfig):
    torch.set_num_threads(2)
    torch.manual_seed(configuration.seed)
    output = Path(configuration.output)
    output.mkdir(parents=True, exist_ok=False)
    metadata = json.loads(
        (
            Path(configuration.artifacts)
            / "motion"
            / configuration.split
            / "clips.json"
        ).read_text()
    )
    if configuration.num_envs % len(metadata["records"]):
        raise ValueError("Evaluation requires equal repeats per clip")
    environment_configuration, agent_configuration = build_configuration(
        configuration, configuration.split, evaluation=True
    )
    if configuration.perturbation:
        amplitude = configuration.perturbation
        environment_configuration.commands["motion"].velocity_range = {
            name: (-amplitude, amplitude)
            for name in ["x", "y", "z", "roll", "pitch", "yaw"]
        }
    environment = TrackingEnvironment(
        cfg=environment_configuration, device=configuration.device
    )
    wrapper = make_wrapper(environment, configuration, agent_configuration.clip_actions)
    runner = MjlabOnPolicyRunner(
        wrapper, dataclasses.asdict(agent_configuration), device=configuration.device
    )
    runner.load(configuration.checkpoint, map_location=configuration.device)
    policy = runner.get_inference_policy(device=configuration.device)
    # Initialization of the runner consumes randomness. Reset it before paired
    # environment resets so perturbations are identical across architectures.
    torch.manual_seed(configuration.seed)
    observations, _ = wrapper.reset()
    command = environment.command_manager.get_term("motion")
    clip_ids = command.clip_ids.clone()
    lengths = command.ends[clip_ids] - command.starts[clip_ids]
    active = torch.ones(
        configuration.num_envs, dtype=torch.bool, device=configuration.device
    )
    completed = torch.zeros_like(active)
    counts = torch.zeros(configuration.num_envs, device=configuration.device)
    error_sums = {
        name: torch.zeros_like(counts)
        for name in ["root_m", "body_m", "joint_rad", "action_delta_squared"]
    }
    failure_names = [[] for _ in range(configuration.num_envs)]
    previous_actions = torch.zeros(
        configuration.num_envs, 29, device=configuration.device
    )
    trajectories = []
    max_frames = int(lengths.max()) + 2
    for frame_index in range(max_frames):
        if not active.any():
            break
        with torch.inference_mode():
            actions = policy(observations)
            physical_actions = (
                wrapper.to_environment_actions(actions)
                if hasattr(wrapper, "to_environment_actions")
                else actions
            )
        errors = {
            "root_m": (command.anchor_pos_w - command.robot_anchor_pos_w).norm(dim=-1),
            "body_m": (command.body_pos_w - command.robot_body_pos_w)
            .norm(dim=-1)
            .mean(dim=-1),
            "joint_rad": (command.joint_pos - command.robot_joint_pos)
            .square()
            .mean(dim=-1)
            .sqrt(),
            "action_delta_squared": (physical_actions - previous_actions)
            .square()
            .mean(dim=-1),
        }
        for name, values in errors.items():
            error_sums[name] += values * active
        counts += active
        robot = environment.scene["robot"]
        trajectories.append(
            torch.cat(
                [
                    robot.data.root_link_pos_w - environment.scene.env_origins,
                    robot.data.root_link_quat_w,
                    robot.data.joint_pos,
                ],
                dim=-1,
            )
            .cpu()
            .numpy()
        )
        observations, _, dones, _ = wrapper.step(actions)
        newly_done = active & dones.bool()
        reference_end = environment.termination_manager.get_term("reference_end")
        failures = torch.zeros_like(active)
        for name in ["anchor_pos", "anchor_ori", "ee_body_pos"]:
            term = environment.termination_manager.get_term(name)
            failures |= term
            for index in torch.where(newly_done & term)[0].tolist():
                failure_names[index].append(name)
        completed |= newly_done & reference_end & ~failures
        active &= ~newly_done
        previous_actions = physical_actions
    rows = []
    for index in range(configuration.num_envs):
        record = metadata["records"][int(clip_ids[index])]
        values = {
            name: float(total[index] / counts[index].clamp_min(1))
            for name, total in error_sums.items()
        }
        for name, cap in [("root_m", 0.5), ("body_m", 0.5), ("joint_rad", 1.0)]:
            missing = (lengths[index] - counts[index]).clamp_min(0)
            values[name + "_failure_penalized"] = float(
                (error_sums[name][index] + missing * cap) / lengths[index]
            )
        rows.append(
            {
                "task": record["task"],
                "motion_seed": record["seed"],
                "split": record["split"],
                "complete": bool(completed[index]),
                "tracking_success": bool(completed[index])
                and values["root_m"] < 0.15
                and values["body_m"] < 0.20
                and values["joint_rad"] < 0.40,
                "frames": int(counts[index]),
                "expected_frames": int(lengths[index]),
                "failure": failure_names[index],
                **values,
            }
        )
    tasks = sorted({row["task"] for row in rows})
    per_task = {}
    numeric_fields = [
        "complete",
        "tracking_success",
        "root_m",
        "body_m",
        "joint_rad",
        "root_m_failure_penalized",
        "body_m_failure_penalized",
        "joint_rad_failure_penalized",
        "action_delta_squared",
    ]
    for task in tasks:
        selected = [row for row in rows if row["task"] == task]
        per_task[task] = {
            name: float(np.mean([row[name] for row in selected]))
            for name in numeric_fields
        }
    summary = {
        name: float(np.mean([values[name] for values in per_task.values()]))
        for name in numeric_fields
    }
    result = {
        "configuration": OmegaConf.to_container(configuration, resolve=True),
        "checkpoint_sha256": hashlib.sha256(
            Path(configuration.checkpoint).read_bytes()
        ).hexdigest(),
        "macro": summary,
        "per_task": per_task,
        "episodes": rows,
        "limitations": "Error averages include initial state; failed horizons use fixed documented penalties. Completion is physical tracking termination success, not prompt semantic success.",
    }
    (output / "metrics.json").write_text(json.dumps(result, indent=2))
    np.savez_compressed(
        output / "rollouts.npz",
        states=np.asarray(trajectories),
        counts=counts.cpu().numpy(),
        clip_ids=clip_ids.cpu().numpy(),
    )
    print("EVALUATION_COMPLETE", json.dumps(summary), flush=True)
    environment.close()


if __name__ == "__main__":
    main()
