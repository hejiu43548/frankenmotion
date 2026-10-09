"""Run independent stage-two diagnostics with an audited, frozen trainer."""

import fcntl
import json
import os
from pathlib import Path
import subprocess
import traceback

import hydra
from omegaconf import OmegaConf


def save_state(path, state):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    temporary.replace(path)


@hydra.main(
    version_base="1.3", config_path="../config", config_name="single_task_stage2"
)
def main(config):
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    with (output / "launcher.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        if (output / "launch.json").exists():
            raise FileExistsError(
                "Existing run: preserve it and resume individual tasks explicitly"
            )
        OmegaConf.save(config, output / "experiment.yaml", resolve=True)
        state = dict(
            state="starting", pid=os.getpid(), completed=[], active=None, stage=2
        )
        save_state(output / "state.json", state)
        commands = {}
        for task_name in config.tasks:
            commands[task_name] = [
                config.python,
                str(Path(config.code_root) / "scripts/train.py"),
                "stage=stage2",
                "controller=with_root",
                f"experiment=single_{task_name}",
                "run_date=20261010",
                f"seed={config.seed}",
                f"initial={config.initial_root}",
                f"backbone.checkpoint={config.backbone}",
                f"data.train_manifest={config.dataset_directory}/train.json",
                f"data.val_manifest={config.dataset_directory}/val.json",
                f"data.path_root={config.cache_root}",
                f"data.skeleton={config.skeleton}",
                f"data.tasks=[{task_name}]",
                f"stage.steps={config.steps}",
                f"stage.batch_size={config.batch_size}",
                f"stage.eval_every={config.eval_every}",
                "validation.points=10",
                "validation.ddim_steps=50",
                f"paths.report={output}/{task_name}/reports",
                f"paths.artifacts={output}/{task_name}/artifacts",
            ]
        save_state(
            output / "launch.json",
            dict(commands=commands, cwd=config.code_root, stage=2),
        )
        try:
            for task_name, command in commands.items():
                state.update(state="training", active=task_name)
                task_directory = output / task_name
                task_directory.mkdir(exist_ok=True)
                with (task_directory / "train.log").open("w") as log:
                    process = subprocess.Popen(
                        command,
                        cwd=config.code_root,
                        stdout=log,
                        stderr=subprocess.STDOUT,
                    )
                    state["child_pid"] = process.pid
                    save_state(output / "state.json", state)
                    return_code = process.wait()
                if return_code:
                    raise RuntimeError(f"{task_name} failed with exit {return_code}")
                report = json.loads(
                    (task_directory / "reports/status.json").read_text()
                )
                if report["state"] != "complete" or report["step"] != config.steps:
                    raise RuntimeError(f"{task_name} did not finish its stage-two plan")
                state["completed"].append(task_name)
            state.update(state="complete", active=None, child_pid=None)
        except BaseException:
            state.update(state="failed", error=traceback.format_exc())
            raise
        finally:
            save_state(output / "state.json", state)


if __name__ == "__main__":
    main()
