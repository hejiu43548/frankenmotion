"""Train a single G1 policy with online PPO and no teacher action targets."""

import dataclasses
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import random
import shutil
import sys
import time

import hydra
import numpy as np
from omegaconf import DictConfig
from omegaconf import OmegaConf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from mjlab.rl import MjlabOnPolicyRunner
from shared_motion.rl.residual import ReferenceResidualWrapper
from shared_motion.rl.environment import build_configuration
from shared_motion.rl.environment import TrackingEnvironment


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def describe(value):
    if callable(value):
        return value.__module__ + "." + value.__qualname__
    return str(value)


@hydra.main(
    config_path="../config/tracker_rl", config_name="residual", version_base="1.3"
)
def main(configuration: DictConfig):
    if configuration.resume is not None:
        raise ValueError(
            "Resume requires a separately audited checkpoint/state path; start a fresh run"
        )
    torch.set_num_threads(2)
    torch.manual_seed(configuration.seed)
    np.random.seed(configuration.seed)
    random.seed(configuration.seed)
    root = Path(__file__).resolve().parents[1]
    report = (
        root
        / configuration.report_root
        / f"{configuration.experiment}_{configuration.run_date}"
    )
    report.mkdir(parents=True, exist_ok=False)
    checkpoints = (
        Path(configuration.artifacts)
        / "training"
        / f"{configuration.experiment}_{configuration.run_date}"
    )
    checkpoints.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(configuration, report / "config.yaml")
    snapshot = report / "source"
    snapshot.mkdir()
    sources = [Path(__file__), *sorted((root / "shared_motion/rl").glob("*.py"))]
    source_hashes = {}
    for source in sources:
        shutil.copy2(source, snapshot / source.name)
        source_hashes[str(source.relative_to(root))] = digest(source)
    environment_configuration, agent_configuration = build_configuration(configuration)
    protocol = {
        "configuration": OmegaConf.to_container(configuration, resolve=True),
        "sources": source_hashes,
        "data_sha256": {
            name: digest(Path(configuration.artifacts) / "motion/train" / name)
            for name in ["motions.npz", "clips.json"]
        },
        "initialization": "random residual policy around kinematic reference joints; no teacher, distilled policy, expert action labels or pretrained checkpoint",
        "rewards": dataclasses.asdict(environment_configuration)["rewards"],
        "runner": dataclasses.asdict(agent_configuration),
        "versions": {
            package: importlib.metadata.version(package)
            for package in [
                "mjlab",
                "rsl-rl-lib",
                "torch",
                "mujoco",
                "mujoco-warp",
                "warp-lang",
            ]
        },
    }
    (report / "protocol.json").write_text(
        json.dumps(protocol, indent=2, default=describe)
    )
    environment = TrackingEnvironment(
        cfg=environment_configuration, device=configuration.device
    )
    wrapper = ReferenceResidualWrapper(
        environment, clip_actions=agent_configuration.clip_actions
    )
    runner = MjlabOnPolicyRunner(
        wrapper,
        dataclasses.asdict(agent_configuration),
        str(checkpoints),
        device=configuration.device,
    )
    original_save = runner.save

    def atomic_save(path, infos=None):
        destination = Path(path)
        temporary = destination.with_suffix(".tmp")
        original_save(str(temporary), infos)
        os.replace(temporary, destination)
        destination.with_suffix(".ready.json").write_text(
            json.dumps(
                {"sha256": digest(destination), "bytes": destination.stat().st_size}
            )
        )

    runner.save = atomic_save
    wrapper.reset()
    command = environment.command_manager.get_term("motion")
    command.time_steps[:] = command.ends[command.clip_ids] - 1
    _, _, dones, _ = wrapper.step(
        torch.zeros(configuration.num_envs, 29, device=configuration.device)
    )
    if not bool(dones.all()) or not bool(
        torch.all(command.time_steps < command.ends[command.clip_ids])
    ):
        raise RuntimeError("Episode-boundary audit failed")
    (report / "boundary_audit.json").write_text(
        json.dumps(
            {"forced_ends": configuration.num_envs, "reported_dones": int(dones.sum())}
        )
    )
    wrapper.reset()
    initial_state = runner.alg.save()
    initial_state.update(iter=0, infos={"initialization": "random"})
    torch.save(initial_state, checkpoints / "initial.pt")
    start_time = time.monotonic()
    runner.learn(
        num_learning_iterations=configuration.iterations, init_at_random_ep_len=False
    )
    (report / "complete.json").write_text(
        json.dumps(
            {
                "iterations": configuration.iterations,
                "wall_seconds": time.monotonic() - start_time,
                "sample_counts": command.sample_counts.tolist(),
                "checkpoints": str(checkpoints),
                "max_torch_memory_bytes": torch.cuda.max_memory_allocated(),
            }
        )
    )
    environment.close()


if __name__ == "__main__":
    main()
