"""Continue this experiment's online PPO with fresh simulator/RNG state.

This is explicitly a new training segment, not a bitwise interrupted-run resume.
Only a checkpoint recorded by a completed local experiment is accepted.
"""

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
from mjlab.rl import RslRlVecEnvWrapper
from shared_motion.rl.environment import build_configuration
from shared_motion.rl.environment import TrackingEnvironment
from shared_motion.rl.residual import ReferenceResidualWrapper


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def assert_restored(expected, actual, location="optimizer"):
    if isinstance(expected, torch.Tensor):
        equal = torch.equal(expected, actual)
    elif isinstance(expected, dict):
        if expected.keys() != actual.keys():
            raise RuntimeError(f"State keys differ: {location}")
        for key in expected:
            assert_restored(expected[key], actual[key], f"{location}.{key}")
        return
    elif isinstance(expected, (list, tuple)):
        if len(expected) != len(actual):
            raise RuntimeError(f"State lengths differ: {location}")
        for index, value in enumerate(expected):
            assert_restored(value, actual[index], f"{location}.{index}")
        return
    else:
        equal = expected == actual
    if not equal:
        raise RuntimeError(f"State restoration mismatch: {location}")


@hydra.main(
    config_path="../config/tracker_rl", config_name="continue", version_base="1.3"
)
def main(arguments: DictConfig):
    root = Path(__file__).resolve().parents[1]
    parent = Path(arguments.parent_report)
    protocol = json.loads((parent / "protocol.json").read_text())
    completed = json.loads((parent / "complete.json").read_text())
    for package, expected in protocol["versions"].items():
        if importlib.metadata.version(package) != expected:
            raise ValueError(f"Parent dependency version changed: {package}")
    checkpoint = Path(arguments.checkpoint).resolve()
    if checkpoint.parent != Path(completed["checkpoints"]).resolve():
        raise ValueError("Checkpoint must belong to the declared parent run")
    ready = json.loads(checkpoint.with_suffix(".ready.json").read_text())
    if digest(checkpoint) != ready["sha256"]:
        raise ValueError("Checkpoint digest differs from recorded ready marker")
    configuration = OmegaConf.load(parent / "config.yaml")
    parent_num_envs = configuration.num_envs
    if arguments.num_envs is not None:
        if arguments.num_envs < 20 or arguments.num_envs % 20:
            raise ValueError("num_envs must give every task an equal number of slots")
        configuration.num_envs = arguments.num_envs
    for name in ["experiment", "seed", "iterations"]:
        configuration[name] = arguments[name]
    configuration.resume = str(checkpoint)
    for name, expected in protocol["data_sha256"].items():
        if digest(Path(configuration.artifacts) / "motion/train" / name) != expected:
            raise ValueError(f"Parent training data changed: {name}")
    for name in ["environment.py", "motion.py", "residual.py"]:
        source = "shared_motion/rl/" + name
        if (
            source in protocol["sources"]
            and digest(root / source) != protocol["sources"][source]
        ):
            raise ValueError(f"Parent training semantics changed: {source}")
    torch.set_num_threads(2)
    torch.manual_seed(configuration.seed)
    np.random.seed(configuration.seed)
    random.seed(configuration.seed)
    name = f"{configuration.experiment}_{configuration.run_date}"
    report = root / configuration.report_root / name
    report.mkdir(parents=True, exist_ok=False)
    checkpoints = Path(configuration.artifacts) / "training" / name
    checkpoints.mkdir(parents=True, exist_ok=False)
    OmegaConf.save(configuration, report / "config.yaml")
    sources = [Path(__file__), *sorted((root / "shared_motion/rl").glob("*.py"))]
    snapshot = report / "source"
    snapshot.mkdir()
    for source in sources:
        shutil.copy2(source, snapshot / source.name)
    protocol.update(
        configuration=OmegaConf.to_container(configuration, resolve=True),
        sources={str(source.relative_to(root)): digest(source) for source in sources},
        parent_report=str(parent.resolve()),
        parent_checkpoint_sha256=ready["sha256"],
        parent_num_envs=parent_num_envs,
        initialization="Continue own online PPO actor, critic, normalizers and optimizer; fresh seeded simulator state, not exact interrupted-run resume",
    )
    (report / "protocol.json").write_text(json.dumps(protocol, indent=2))
    environment_configuration, agent_configuration = build_configuration(configuration)
    environment = TrackingEnvironment(
        cfg=environment_configuration, device=configuration.device
    )
    wrapper_type = (
        ReferenceResidualWrapper
        if configuration.get("reference_residual", False)
        else RslRlVecEnvWrapper
    )
    wrapper = wrapper_type(environment, clip_actions=agent_configuration.clip_actions)
    runner = MjlabOnPolicyRunner(
        wrapper,
        dataclasses.asdict(agent_configuration),
        str(checkpoints),
        device=configuration.device,
    )
    runner.load(str(checkpoint), map_location=configuration.device)
    # Upstream stores the last completed zero-based update, not the next update.
    runner.current_learning_iteration += 1
    # Optimizer.load restores its LR but upstream PPO's scalar is not serialized.
    rates = {group["lr"] for group in runner.alg.optimizer.param_groups}
    if len(rates) != 1:
        raise ValueError("Expected one shared optimizer learning rate")
    runner.alg.learning_rate = rates.pop()
    loaded = torch.load(
        checkpoint, map_location=configuration.device, weights_only=False
    )
    restored = runner.alg.save()
    assert_restored(loaded["optimizer_state_dict"], restored["optimizer_state_dict"])
    for state_name in ["actor_state_dict", "critic_state_dict"]:
        for key, expected in loaded[state_name].items():
            if not torch.equal(expected, restored[state_name][key]):
                raise RuntimeError(f"State restoration mismatch: {state_name}.{key}")
    (report / "restoration_audit.json").write_text(
        json.dumps(
            {
                "actor_critic_normalizers_exact": True,
                "optimizer_exact": True,
                "first_update": runner.current_learning_iteration,
                "learning_rate": runner.alg.learning_rate,
                "simulator_state": "fresh reset",
            },
            indent=2,
        )
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
    started = time.monotonic()
    runner.learn(
        num_learning_iterations=configuration.iterations, init_at_random_ep_len=False
    )
    (report / "complete.json").write_text(
        json.dumps(
            {
                "iterations": configuration.iterations,
                "final_update": runner.current_learning_iteration,
                "wall_seconds": time.monotonic() - started,
                "checkpoints": str(checkpoints),
            },
            indent=2,
        )
    )
    environment.close()


if __name__ == "__main__":
    main()
