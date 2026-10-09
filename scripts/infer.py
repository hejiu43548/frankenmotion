#!/usr/bin/env python3
"""Generate from a staged checkpoint; legacy shared20 releases keep their CLI."""
from pathlib import Path
import sys

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@hydra.main(version_base="1.3", config_path="../config", config_name="infer")
def main(config: DictConfig):
    import numpy as np
    import torch

    from shared_motion.training.catalog import TASK_NAMES
    from shared_motion.training.geometry import Skeleton
    from shared_motion.training.model import build_model
    from shared_motion.training.runner import checkpoint_compatible

    checkpoint = torch.load(config.checkpoint, map_location="cpu", weights_only=False)
    if checkpoint["stage"] not in [2, 3] or config.task not in checkpoint["tasks"]:
        raise ValueError("Choose a stage2/3 checkpoint and a task trained by it")
    model = build_model(config, "task").to(config.device).eval()
    checkpoint_compatible(checkpoint, model, checkpoint["tasks"])
    model.load_adapter(checkpoint["adapter"])
    skeleton = Skeleton(config.skeleton).to(config.device)
    with np.load(config.text_cache, allow_pickle=False) as archive:
        local = torch.tensor(
            archive["local"], dtype=torch.float32, device=config.device
        )[None]
        local_mask = torch.tensor(archive["local_mask"], device=config.device)[None]
        text = torch.tensor(archive["tx"], dtype=torch.float32, device=config.device)[
            None
        ]
    if local.shape[-1] != 408 or text.shape != (1, 512):
        raise ValueError("Expected cached local408 and tx512 text features")
    frames = local.shape[1]
    batch = dict(
        local=local,
        local_mask=local_mask,
        tx=text,
        lengths=torch.tensor([frames], device=config.device),
        mask=torch.ones(1, frames, dtype=torch.bool, device=config.device),
        task=torch.tensor([TASK_NAMES.index(config.task)], device=config.device),
    )
    with torch.no_grad():
        motion = model.sample(
            batch,
            torch.tensor([config.command], dtype=torch.float32, device=config.device),
            skeleton,
            [config.seed],
            config.ddim_steps,
        )
    destination = Path(config.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        destination,
        motion=motion[0].cpu().numpy(),
        fps=20.0,
        task=config.task,
        command=config.command,
        seed=config.seed,
    )
    print(destination)


if __name__ == "__main__":
    main()
