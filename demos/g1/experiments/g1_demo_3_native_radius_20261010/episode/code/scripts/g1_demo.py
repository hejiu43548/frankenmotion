"""Reproducible continuous G1 interaction demo; artifacts retain every layer."""

import hashlib
import json
from pathlib import Path
import sys

import hydra
import numpy as np
from omegaconf import OmegaConf


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def generate(config):
    import torch
    from scipy.spatial.transform import Rotation

    sys.path.insert(0, config.frozen_code)
    from shared_motion.training.catalog import TASK_NAMES
    from shared_motion.training.geometry import Skeleton
    from shared_motion.training.model import build_model
    from shared_motion.training.runner import checkpoint_compatible
    from src.tools.geometry import rotation_6d_to_matrix, matrix_to_euler_angles
    from src.tools.geometry import axis_angle_rotation

    torch.set_num_threads(8)
    device = str(config.generation_device)
    assert digest(config.checkpoint) == config.checkpoint_sha256
    training = OmegaConf.load(config.training_config)
    model = build_model(training, "task").to(device).eval()
    checkpoint = torch.load(config.checkpoint, map_location="cpu", weights_only=False)
    checkpoint_compatible(checkpoint, model, checkpoint["tasks"])
    model.load_adapter(checkpoint["adapter"])
    skeleton = Skeleton(training.data.skeleton).to(device)
    rows = json.loads(Path(config.manifest).read_text())
    destination = Path(config.output) / "generated"
    destination.mkdir(parents=True, exist_ok=True)
    records = []
    for segment in config.segments:
        row = next(
            row
            for row in rows
            if row["task"] == segment.task and row["key"] == segment.key
        )
        archive = np.load(row["cache"])
        count = len(config.seeds)
        frames = len(archive["local"])
        batch = {
            name: torch.tensor(archive[name], device=device)[None].repeat(
                count, *([1] * archive[name].ndim)
            )
            for name in ["local", "local_mask", "tx"]
        }
        batch.update(
            lengths=torch.full((count,), frames, device=device),
            mask=torch.ones(count, frames, device=device, dtype=torch.bool),
            task=torch.full((count,), TASK_NAMES.index(segment.task), device=device),
        )
        with torch.inference_mode():
            motion = model.sample(
                batch,
                torch.full((count,), segment.command, device=device),
                skeleton,
                list(config.seeds),
                50,
            )
            rotations = rotation_6d_to_matrix(
                motion[..., 4:136].reshape(count, frames, 22, 6)
            )
            angles = matrix_to_euler_angles(rotations[:, :, 0], "ZYX")
            yaw = torch.cat(
                [torch.zeros_like(motion[:, :1, 3]), motion[:, :-1, 3].cumsum(1)], 1
            )
            heading_rotation = axis_angle_rotation("Z", yaw)
            rotations[:, :, 0] = (
                heading_rotation
                @ axis_angle_rotation("Y", angles[..., 1])
                @ axis_angle_rotation("X", angles[..., 2])
            )
            velocity = (heading_rotation[..., :2, :2] @ motion[..., 1:3, None]).squeeze(
                -1
            )
            root_xy = torch.cat(
                [torch.zeros_like(velocity[:, :1]), velocity[:, :-1].cumsum(1)], 1
            )
            positions = [torch.cat([root_xy, motion[..., :1]], -1)]
            global_rotations = [rotations[:, :, 0]]
            for joint_index in range(1, 22):
                parent = skeleton.parents[joint_index]
                global_rotations.append(
                    global_rotations[parent] @ rotations[:, :, joint_index]
                )
                offset = (
                    skeleton.rest_positions[joint_index]
                    - skeleton.rest_positions[parent]
                )
                positions.append(
                    positions[parent]
                    + (global_rotations[parent] @ offset[:, None]).squeeze(-1)
                )
            positions = torch.stack(positions, 2).cpu().numpy()
            global_rotations = torch.stack(global_rotations, 2).cpu().numpy()
        for seed_index, seed in enumerate(config.seeds):
            path = destination / f"{segment.task}_{seed}.npz"
            np.savez_compressed(
                path,
                motion=motion[seed_index].cpu().numpy(),
                positions=positions[seed_index],
                quaternions=Rotation.from_matrix(
                    global_rotations[seed_index].reshape(-1, 3, 3)
                )
                .as_quat()
                .reshape(frames, 22, 4)[..., [3, 0, 1, 2]],
                fps=20.0,
                human_height=skeleton.height,
            )
            records.append(
                dict(
                    task=segment.task,
                    seed=seed,
                    command=segment.command,
                    frames=frames,
                    caption=row["caption"],
                    text_cache=row["cache"],
                    text_cache_sha256=digest(row["cache"]),
                    file=str(path),
                    sha256=digest(path),
                    checkpoint_sha256=config.checkpoint_sha256,
                    generation_device=device,
                    torch_version=torch.__version__,
                )
            )
        print(segment.task, "generated", frames, "frames", flush=True)
    (destination / "manifest.json").write_text(json.dumps(records, indent=2))
    OmegaConf.save(config, Path(config.output) / "config.yaml")


@hydra.main(version_base="1.3", config_path="../config", config_name="g1_demo")
def main(config):
    if config.phase == "generate":
        generate(config)
    else:
        from g1_demo_physics import run_phase

        run_phase(config)


if __name__ == "__main__":
    main()
