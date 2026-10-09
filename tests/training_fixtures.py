"""Small deterministic fixtures, never production motion or pretrained weights."""

import json
import hashlib
from pathlib import Path

import numpy as np
import torch
from torch import nn

from shared_motion.training.catalog import TASK_NAMES
from shared_motion.training.turn import TURN_REVISION


class ToyEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = nn.ModuleList([nn.Identity() for _ in range(4)])

    def forward(self, features):
        for layer in self.layers:
            features = layer(features)
        return features


class ToyBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.latent_dim = 512
        self.input_projection = nn.Linear(613, 512)
        self.seqTransEncoder = ToyEncoder()
        self.output_projection = nn.Linear(512, 613)
        for parameter in self.parameters():
            nn.init.normal_(parameter, std=0.003)

    def forward(self, motion, conditioning, timesteps, final_timesteps=None):
        return self.output_projection(
            self.seqTransEncoder(self.input_projection(motion))
        )


def toy_bundle():
    with torch.random.fork_rng():
        torch.manual_seed(912)
        backbone = ToyBackbone().requires_grad_(False)
    mean = torch.zeros(613)
    mean[0] = 0.9
    mean[4:136] = torch.tensor([1, 0, 0, 0, 1, 0]).repeat(22)
    return dict(
        denoiser=backbone,
        mean=mean,
        std=torch.ones(613),
        text_mean=torch.zeros(512),
        text_std=torch.ones(512),
        alphas=torch.linspace(0.999, 0.001, 100),
        sha256="synthetic-test-backbone",
    )


def make_data(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    positions = np.zeros((38, 3), np.float32)
    for joint_index in range(1, 38):
        positions[joint_index] = [
            (joint_index % 3 - 1) * 0.1,
            (joint_index % 2 * 2 - 1) * 0.1,
            -0.02 * joint_index,
        ]
    parents = np.zeros(38, np.int64)
    parents[0] = -1
    np.savez(
        root / "skeleton.npz",
        J=positions,
        parents=parents,
        height=np.float32(1.372592926),
    )
    for split in ["train", "val"]:
        rows = []
        for task_index, name in enumerate(TASK_NAMES):
            motion = np.zeros((120, 205), np.float32)
            motion[:, 0] = 0.9
            motion[:, 4:136] = np.tile([1, 0, 0, 0, 1, 0], 22)
            motion[:, 1] = 0.01
            motion[:, 3] = -0.5 / 119 if name == "turn" else 0.001
            path = root / f"{split}_{name}.npz"
            np.savez(
                path,
                motion=motion,
                local=np.zeros((120, 408), np.float32),
                local_mask=np.ones((120, 408), bool),
                tx=np.zeros(512, np.float32),
                quantity=np.float32(0.5),
            )
            rows.append(
                dict(
                    task=name,
                    split=split,
                    family=f"synthetic_{split}_{name}",
                    key=name,
                    cache=str(path),
                    target_frames=120,
                )
            )
        turn = next(row for row in rows if row["task"] == "turn")
        turn.update(
            turn_source_revision=TURN_REVISION,
            ready_for_training=True,
            pad_frames=0,
            real_frames=120,
            crop_start_frame_20fps=0,
            crop_end_frame_20fps=120,
            admission_reasons=[],
            direction="right",
            caption="A synthetic right walking turn",
            quantity=0.5,
            cache_sha256=hashlib.sha256(Path(turn["cache"]).read_bytes()).hexdigest(),
        )
        with np.load(turn["cache"]) as archive:
            negative = {name: archive[name].copy() for name in archive.files}
        negative["quantity"] = np.float32(-0.5)
        negative["motion"][:, 3] = -negative["motion"][:, 3]
        negative_path = root / f"{split}_turn_left.npz"
        np.savez(negative_path, **negative)
        rows.append(
            dict(
                turn,
                direction="left",
                caption="A synthetic left walking turn",
                quantity=-0.5,
                key="turn_left",
                family=f"synthetic_{split}_turn_left",
                cache=str(negative_path),
                cache_sha256=hashlib.sha256(negative_path.read_bytes()).hexdigest(),
            )
        )
        (root / f"{split}.json").write_text(json.dumps(rows))
    return root
