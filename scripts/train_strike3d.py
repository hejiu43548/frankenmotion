"""Train newly initialized hand/XYZ residuals on selected real strike motions."""

import json
from pathlib import Path
import random
import sys
import time

import hydra
from hydra.utils import instantiate
import numpy as np
from omegaconf import OmegaConf
import torch
from torch.nn import functional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared_motion.training.geometry import Skeleton
from shared_motion.training.model import file_sha256
from shared_motion.training.reach3d_model import ReachDiffusion


def load_items(directory, split):
    rows = json.loads((Path(directory) / f"{split}.json").read_text())
    items = []
    for row in rows:
        cache_path = Path(row["cache"])
        if not cache_path.is_file():
            cache_path = Path(directory) / "cache" / cache_path.name
        if file_sha256(cache_path) != row["rebuilt_cache_sha256"]:
            raise ValueError("Rebuilt source cache hash mismatch")
        with np.load(cache_path, allow_pickle=False) as archive:
            items.append(
                {
                    name: torch.from_numpy(np.asarray(archive[name]).copy())
                    for name in [
                        "motion",
                        "local",
                        "local_mask",
                        "tx",
                        "hands",
                        "positions",
                        "event_frames",
                    ]
                }
            )
    for item in items:
        for name in ["motion", "local", "tx", "positions"]:
            item[name] = item[name].float()
    return rows, items


def make_batch(items, indices, device):
    selected = [items[index] for index in indices]
    lengths = torch.tensor([len(item["motion"]) for item in selected], device=device)
    frames = int(lengths.max())
    batch = {
        name: torch.stack(
            [
                functional.pad(item[name], (0, 0, 0, frames - len(item[name])))
                for item in selected
            ]
        ).to(device)
        for name in ["motion", "local", "local_mask"]
    }
    for name in ["tx", "hands", "positions", "event_frames"]:
        batch[name] = torch.stack([item[name] for item in selected]).to(device)
    batch["mask"] = torch.arange(frames, device=device)[None] < lengths[:, None]
    return batch


@hydra.main(version_base="1.3", config_path="../config", config_name="train_strike3d")
def main(config):
    torch.set_num_threads(4)
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    torch.cuda.manual_seed_all(config.seed)
    output = Path(config.output)
    if (output / "final.pt").exists() or (output / "metrics.jsonl").exists():
        raise ValueError("Refusing to overwrite an existing training run")
    output.mkdir(parents=True, exist_ok=True)
    OmegaConf.save(config, output / "config.yaml", resolve=True)
    rows, items = load_items(config.data, "train")
    validation_rows, _ = load_items(config.data, "val")
    if {row["family"] for row in rows} & {row["family"] for row in validation_rows}:
        raise ValueError("Source family leakage")
    audit = json.loads((Path(config.data) / "audit.json").read_text())
    target_frame = config.get("target_frame", "instantaneous_body")
    checkpoint_format = (
        "fixed_world_adapter_v1"
        if target_frame == "fixed_world"
        else "strike_xyz_residual_v1"
    )
    if audit.get("target_frame", "instantaneous_body") != target_frame:
        raise ValueError("Dataset target frame does not match training")
    model = ReachDiffusion(
        instantiate(config.backbone),
        audit["train_target_center"],
        audit["train_target_scale"],
    ).to(config.device)
    skeleton = Skeleton(config.skeleton).to(config.device)
    initial_sha256 = None
    if config.initial:
        initial = torch.load(config.initial, map_location="cpu", weights_only=False)
        if (
            initial.get("format") != checkpoint_format
            or initial["backbone_sha256"] != model.backbone_sha256
        ):
            raise ValueError("Expected this experiment's strike residual checkpoint")
        if initial["train_manifest_sha256"] != file_sha256(
            Path(config.data) / "train.json"
        ):
            raise ValueError("Continuation dataset mismatch")
        model.load_adapter(initial["adapter"])
        initial_sha256 = file_sha256(config.initial)
    model.train()
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=config.learning_rate,
        weight_decay=0.01,
    )
    generator = torch.Generator(device=config.device).manual_seed(config.seed + 1)
    sampler = np.random.default_rng(config.seed + 2)
    started = time.monotonic()
    for step in range(1, config.steps + 1):
        indices = sampler.integers(0, len(items), size=config.batch_size).tolist()
        batch = make_batch(items, indices, config.device)
        optimizer.zero_grad(set_to_none=True)
        loss, metrics = model.supervised_loss(
            batch, generator, skeleton, config.endpoint_weight, target_frame
        )
        if not torch.isfinite(loss):
            raise FloatingPointError("Nonfinite strike training loss")
        loss.backward()
        gradient_norm = torch.nn.utils.clip_grad_norm_(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            1.0,
            error_if_nonfinite=True,
        )
        optimizer.step()
        if step == 1 or step % config.log_every == 0:
            record = dict(
                step=step,
                loss=float(loss.detach()),
                gradient_norm=float(gradient_norm),
                seconds=time.monotonic() - started,
                **{name: float(value) for name, value in metrics.items()},
            )
            with (output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(record) + "\n")
            print(json.dumps(record), flush=True)
        if step % config.checkpoint_every == 0 or step == config.steps:
            checkpoint = dict(
                format=checkpoint_format,
                initial_sha256=initial_sha256,
                step=step,
                adapter=model.adapter_state(),
                backbone_sha256=model.backbone_sha256,
                optimizer=optimizer.state_dict(),
                generator_state=generator.get_state(),
                sampler_state=sampler.bit_generator.state,
                torch_rng=torch.get_rng_state(),
                cuda_rng=torch.cuda.get_rng_state_all(),
                config=OmegaConf.to_container(config, resolve=True),
                train_manifest_sha256=file_sha256(Path(config.data) / "train.json"),
                audit=audit,
            )
            destination = output / (
                "final.pt" if step == config.steps else f"checkpoint_{step:06d}.pt"
            )
            torch.save(checkpoint, destination.with_suffix(".tmp"))
            destination.with_suffix(".tmp").replace(destination)
    (output / "completion.json").write_text(
        json.dumps(
            dict(
                steps=config.steps,
                seconds=time.monotonic() - started,
                final_sha256=file_sha256(output / "final.pt"),
                validation_used_for_updates=False,
                teacher_used=False,
            ),
            indent=2,
        )
        + "\n"
    )


if __name__ == "__main__":
    main()
