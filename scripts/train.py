#!/usr/bin/env python3
"""Hydra entrypoint for all three human-motion training stages."""
from pathlib import Path
import sys

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@hydra.main(version_base="1.3", config_path="../config", config_name="train")
def main(config: DictConfig):
    from shared_motion.training.runner import run

    run(config)


if __name__ == "__main__":
    main()
