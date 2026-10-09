"""Train online PPO using the shared audited trainer."""

from pathlib import Path
import sys

import hydra
from omegaconf import DictConfig

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared_motion.rl.training import train


@hydra.main(config_path="../config/tracker_rl", config_name="train", version_base="1.3")
def main(configuration: DictConfig):
    train(configuration, Path(__file__).resolve())


if __name__ == "__main__":
    main()
