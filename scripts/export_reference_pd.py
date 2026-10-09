"""Export the untrained reference-position PD baseline for paired evaluation."""

import json
from pathlib import Path
import sys

import hydra
from omegaconf import DictConfig
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


class ReferencePD(torch.nn.Module):
    def __init__(self, contract):
        super().__init__()
        self.register_buffer("scale", torch.tensor(contract["action_scale"]))
        self.register_buffer("offset", torch.tensor(contract["action_offset"]))

    def forward(self, observations):
        return (observations[:, :29] - self.offset) / self.scale


@hydra.main(
    config_path="../config/tracker_rl", config_name="reference_pd", version_base="1.3"
)
def main(configuration: DictConfig):
    contract = json.loads(Path(configuration.contract).read_text())
    output = Path(configuration.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        raise FileExistsError(output)
    policy = ReferencePD(contract).eval()
    torch.jit.script(policy).save(str(output))


if __name__ == "__main__":
    main()
