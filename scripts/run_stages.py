#!/usr/bin/env python3
"""One locked supervisor for the configured stage2 -> stage3 sequence."""

import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import hydra
from omegaconf import OmegaConf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared_motion.training.model import file_sha256
from shared_motion.training.runner import save_json


@hydra.main(version_base="1.3", config_path="../config", config_name="pipeline")
def main(config):
    output = Path(config.output).resolve()
    output.mkdir(parents=True, exist_ok=True)
    with (output / "pipeline.lock").open("w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        state_path = output / "pipeline_state.json"
        if state_path.exists() and not config.resume:
            raise FileExistsError(
                "Pipeline exists; inspect processes and explicitly set resume=true"
            )
        state = json.loads(state_path.read_text()) if state_path.exists() else {}
        if config.resume:
            for stage in [2, 3]:
                pid = state.get(f"stage{stage}_pid")
                if pid and Path(f"/proc/{pid}").exists():
                    raise RuntimeError(
                        f"Refusing to duplicate live recorded process {pid}"
                    )
        state.update(supervisor_pid=os.getpid(), state="running")
        save_json(state_path, state)
        OmegaConf.save(config, output / "pipeline_config.yaml", resolve=True)
        try:
            import torch

            if not torch.cuda.is_available():
                raise RuntimeError("CUDA is unavailable; no training process launched")
            initial = Path(config.initial_root)
            for stage in [2, 3]:
                report = output / "reports" / f"stage{stage}"
                artifacts = output / f"stage{stage}"
                status_path = report / "status.json"
                status = (
                    json.loads(status_path.read_text()) if status_path.exists() else {}
                )
                state.update(active_stage=stage, active_run=str(report))
                state[f"stage{stage}_run"] = str(report)
                state[f"stage{stage}_artifacts"] = str(artifacts)
                if status.get("state") != "complete":
                    command = [
                        sys.executable,
                        "scripts/train.py",
                        f"stage=stage{stage}",
                        f"controller={config.controller}",
                        f"experiment={config.experiment}",
                        f"run_date={config.run_date}",
                        f"backbone.checkpoint={config.backbone}",
                        f"data.train_manifest={config.train_manifest}",
                        f"data.val_manifest={config.val_manifest}",
                        f"data.path_root={config.cache_root}",
                        f"data.skeleton={config.skeleton}",
                        f"initial={initial}",
                        f"paths.report={report}",
                        f"paths.artifacts={artifacts}",
                    ]
                    command.extend(
                        str(value) for value in config[f"stage{stage}_overrides"]
                    )
                    if (report / "protocol.json").exists():
                        if not config.resume or not (artifacts / "latest.pt").exists():
                            raise RuntimeError(
                                "Partially initialized stage without a resumable checkpoint; preserve evidence"
                            )
                        command.append(f"runtime.resume={artifacts / 'latest.pt'}")
                    state[f"stage{stage}_command"] = command
                    save_json(state_path, state)
                    with (output / f"stage{stage}.log").open("a") as log:
                        child = subprocess.Popen(
                            command,
                            stdout=log,
                            stderr=subprocess.STDOUT,
                            cwd=Path(__file__).resolve().parents[1],
                        )
                        state[f"stage{stage}_pid"] = child.pid
                        save_json(state_path, state)
                        result = child.wait()
                    if result:
                        raise RuntimeError(
                            f"Stage{stage} exited {result}; inspect its log/failure.json"
                        )
                status = json.loads(status_path.read_text())
                audit = json.loads((report / "completion_audit.json").read_text())
                if (
                    status["state"] != "complete"
                    or not audit["frozen_parameters_verified"]
                ):
                    raise RuntimeError("Stage completion/freeze audit missing")
                best = json.loads((report / "best.json").read_text())
                initial = Path(best["checkpoint"])
                if file_sha256(initial) != best["sha256"]:
                    raise RuntimeError("Selected checkpoint hash mismatch")
                state[f"stage{stage}_best"] = best
                state[f"stage{stage}_finished_at"] = time.time()
                if stage == 2:
                    save_json(
                        output / "stage3_selection.json",
                        dict(
                            stage2_run=str(report),
                            checkpoint=str(initial),
                            checkpoint_sha256=best["sha256"],
                            step=best["step"],
                            metric=best["metric"],
                            reason="Minimum fixed200-rollout normalized macro MAE across audited stage2 checkpoints",
                            completion_audit=audit,
                        ),
                    )
                save_json(state_path, state)
            state.update(
                state="complete", active_stage="complete", finished_at=time.time()
            )
            save_json(state_path, state)
        except BaseException as error:
            state.update(state="failed", error=repr(error))
            save_json(state_path, state)
            save_json(
                output / "pipeline_failure.json",
                dict(error=repr(error), traceback=traceback.format_exc()),
            )
            raise


if __name__ == "__main__":
    main()
