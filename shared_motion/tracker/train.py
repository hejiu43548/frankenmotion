"""Task-balanced fine tuning of the ONE exported shared correction head."""

import argparse
import copy
import json
import random
from pathlib import Path
import numpy as np
import torch
from shared_motion.data import read_manifest, BalancedTaskSampler


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--retain", type=float, default=1.0)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    rows = read_manifest(args.manifest)
    tracker_model = torch.jit.load(args.initial, map_location=args.device).eval()
    teacher = torch.jit.load(args.initial, map_location=args.device).eval()
    for parameter in tracker_model.parameters():
        parameter.requires_grad_(False)
    for parameter in tracker_model.head.parameters():
        parameter.requires_grad_(True)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    optimizer = torch.optim.Adam(tracker_model.head.parameters(), lr=args.lr)
    cache = {}
    sampler = BalancedTaskSampler(rows, args.steps * args.batch_size, args.seed)
    sample_indices = iter(sampler)

    def sample(row):
        path = row["path"]
        if path not in cache:
            training_data = np.load(path)
            input_features = np.c_[
                training_data["observations"],
                training_data["sonic_actions"],
                training_data["latents"],
            ]
            target_actions = training_data["actions"]
            usable_frames = (
                len(target_actions)
                if row.get("complete", True)
                else max(0, len(target_actions) - 75)
            )
            if usable_frames == 0:
                raise ValueError(f"No usable frames: {path}")
            if input_features.shape != (
                len(target_actions),
                588,
            ) or target_actions.shape[1:] != (29,):
                raise ValueError(f"Invalid teacher dataset: {path}")
            cache[path] = (
                input_features[:usable_frames].astype("float32"),
                target_actions[:usable_frames].astype("float32"),
            )
        input_features, target_actions = cache[path]
        frame_index = rng.randrange(len(target_actions))
        return input_features[frame_index], target_actions[frame_index]

    for step in range(args.steps):
        batch = [sample(rows[next(sample_indices)]) for _ in range(args.batch_size)]
        input_features = torch.as_tensor(
            np.stack([sample_record[0] for sample_record in batch]), device=args.device
        )
        target_actions = torch.as_tensor(
            np.stack([sample_record[1] for sample_record in batch]), device=args.device
        )
        normalized = torch.clamp(
            (input_features - tracker_model.mean) / tracker_model.std, -10, 10
        )
        predicted_actions = input_features[:, 495:524] + 3 * torch.tanh(
            tracker_model.head(normalized)
        )
        with torch.no_grad():
            teacher_actions = input_features[:, 495:524] + 3 * torch.tanh(
                teacher.head(
                    torch.clamp((input_features - teacher.mean) / teacher.std, -10, 10)
                )
            )
        loss = (predicted_actions - target_actions).square().mean() + args.retain * (
            predicted_actions - teacher_actions
        ).square().mean()
        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(tracker_model.head.parameters(), 2.0)
        optimizer.step()
        if step % 100 == 0:
            print(step, float(loss), flush=True)
    tracker_model.reset()
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(tracker_model.cpu(), str(destination))
    checkpoint_metadata = json.loads(
        Path(args.initial).with_suffix(".json").read_text()
    )
    checkpoint_metadata.update(
        training_manifest=str(Path(args.manifest).resolve()),
        sampling="Uniform task cycles; uniform clip then frame; no corpus truncation",
        steps=args.steps,
    )
    destination.with_suffix(".json").write_text(
        json.dumps(checkpoint_metadata, indent=2)
    )


if __name__ == "__main__":
    main()
