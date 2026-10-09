#!/usr/bin/env python3
"""Replace legacy turn rows with audited walking-turn event caches."""

from pathlib import Path
import sys

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


@hydra.main(version_base="1.3", config_path="../config", config_name="repair_turn_data")
def main(config: DictConfig):
    from shared_motion.training.turn_data import repair

    repair(config)


if __name__ == "__main__":
    main()
