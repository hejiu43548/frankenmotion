"""Joint task-balanced training through the deployed DDIM sampler."""

import argparse
import json
import random
from pathlib import Path
import numpy as np
import torch
from shared_motion.data import read_manifest, BalancedTaskSampler
from .model import load, save, command, TASKS
from .schema import RANGES, TASKS as ORIGINAL_TASKS
from .catalog import NEW, quantities
from .kinematics import FK, quantity
from .sampling import generate


def differentiable(model, prompt_cache, command_features, seed):
    local = prompt_cache["local"][None]
    batch_size, num_frames, _ = local.shape
    device = local.device
    local = model.motion_normalizer(
        torch.cat([torch.zeros(batch_size, num_frames, 205, device=device), local], -1)
    )[..., 205:]
    conditioning = dict(
        mask=torch.ones(batch_size, num_frames, device=device, dtype=torch.bool),
        tx=model.prepare_tx_emb(prompt_cache["tx"]),
    )
    noise = torch.randn(
        batch_size,
        num_frames,
        205,
        generator=torch.Generator(device=device).manual_seed(seed),
        device=device,
    )
    steps = np.linspace(model.timesteps - 1, 0, 50, dtype=int)
    model.denoiser.static_residuals = model.denoiser.controller(command_features)
    try:
        for step_index, timestep in enumerate(steps):
            noisy_motion = torch.cat([noise, local], -1)
            prediction = model.denoiser(
                noisy_motion,
                conditioning,
                torch.full(
                    (batch_size,), int(timestep), device=device, dtype=torch.long
                ),
            )
            if step_index == len(steps) - 1:
                return model.motion_normalizer.inverse(prediction)[..., :205]
            alpha = model.alphas_cumprod[timestep]
            next_alpha = model.alphas_cumprod[steps[step_index + 1]]
            noise = (
                next_alpha.sqrt() * prediction
                + (1 - next_alpha).sqrt()
                * (noisy_motion - alpha.sqrt() * prediction)
                / (1 - alpha).sqrt()
            )[..., :205]
    finally:
        model.denoiser.static_residuals = None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--initial", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--skeleton", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--steps", type=int, default=2700)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--lr", type=float, default=3e-5)
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    rows = read_manifest(args.manifest)
    if any(
        record["task"] not in TASKS or not record.get("prompt_cache") for record in rows
    ):
        raise ValueError("Training rows need a supported task and prompt_cache")
    model, checkpoint = load(args.initial, args.device)
    teacher, _ = load(args.initial, args.device)
    model.requires_grad_(False)
    model.denoiser.controller.requires_grad_(True)
    teacher.requires_grad_(False)
    forward_kinematics = FK(args.skeleton, args.device)
    optimizer = torch.optim.AdamW(
        model.denoiser.controller.parameters(), lr=args.lr, weight_decay=1e-6
    )
    cache = {}
    logs = []
    for step, index in enumerate(BalancedTaskSampler(rows, args.steps, args.seed), 1):
        row = rows[index]
        task = row["task"]
        path = row["prompt_cache"]
        if path not in cache:
            cache[path] = torch.load(path, map_location=args.device, weights_only=False)
        prompt_cache = cache[path]
        frames = len(prompt_cache["local"])
        lower_bound, upper_bound = (
            NEW[task]["bounds"] if task in NEW else RANGES[ORIGINAL_TASKS.index(task)]
        )
        value = float(row.get("command", rng.uniform(lower_bound, upper_bound)))
        seed = rng.randrange(2**31)
        command_features = command(task, value, frames, args.device)
        with torch.no_grad():
            target = generate(
                teacher,
                prompt_cache["local"],
                prompt_cache["tx"],
                command_features,
                seed,
            )
            target_positions = forward_kinematics(target)
        predicted_motion = differentiable(model, prompt_cache, command_features, seed)
        predicted_positions = forward_kinematics(predicted_motion)
        measured_quantity = (
            quantities(predicted_positions)[task]
            if task in NEW
            else quantity(
                predicted_positions,
                task,
                scale=1.2701193988323212 / forward_kinematics.height,
            )
        )
        quantity_loss = (
            ((measured_quantity - value) / (upper_bound - lower_bound)).square().mean()
        )
        geometry_loss = (
            (predicted_positions - predicted_positions[:, :, :1])
            - (target_positions - target_positions[:, :, :1])
        ).square().mean() + 0.1 * (
            predicted_positions[:, :, 0] - target_positions[:, :, 0]
        ).square().mean()
        velocity_loss = (
            (predicted_positions[:, 1:] - predicted_positions[:, :-1])
            - (target_positions[:, 1:] - target_positions[:, :-1])
        ).square().mean() * 400
        replay = []
        for replay_task in TASKS:
            low, high = (
                NEW[replay_task]["bounds"]
                if replay_task in NEW
                else RANGES[ORIGINAL_TASKS.index(replay_task)]
            )
            replay.append(
                command(replay_task, rng.uniform(low, high), frames, args.device)
            )
        replay_commands = torch.cat(replay, 0)
        with torch.no_grad():
            teacher_residuals, teacher_output = teacher.denoiser.controller(
                replay_commands
            )
        student_residuals, student_output = model.denoiser.controller(replay_commands)
        retention_loss = (student_residuals - teacher_residuals).square().mean() + (
            student_output - teacher_output
        ).square().mean()
        loss = (
            quantity_loss
            + 2 * geometry_loss
            + 0.05 * velocity_loss
            + 300 * retention_loss
        )
        # Preserve the selected stationary support prior for the nine added classes.
        if task in NEW and task not in ["jog", "march"]:
            feet = predicted_positions[:, :, [7, 8]]
            loss = (
                loss
                + 0.4
                * (
                    (
                        predicted_positions[:, 1:, 0, :2]
                        - predicted_positions[:, :-1, 0, :2]
                    )
                    * 20
                )
                .square()
                .mean()
                + 0.1 * ((feet[:, 1:] - feet[:, :-1]) * 20).square().mean()
            )
            if task != "squat":
                ankles = feet.mean(2)
                height = predicted_positions[:, :, 0, 2] - ankles[:, :, 2]
                offset = (predicted_positions[:, :, 0, :2] - ankles[:, :, :2]).norm(
                    dim=-1
                )
                loss = (
                    loss
                    + 20 * torch.relu(0.82 - height).square().mean()
                    + 5 * torch.relu(offset - 0.12).square().mean()
                )
        # Optional full-corpus motion records supply semantic rehearsal, not IK targets.
        if row.get("path") and task in ["clap", "arm_circle"]:
            source = np.load(row["path"])["motion"]
            sample = np.linspace(0, len(source) - 1, frames)
            source = np.stack(
                [
                    np.interp(sample, np.arange(len(source)), feature_values)
                    for feature_values in source.T
                ],
                1,
            ).astype("float32")
            with torch.no_grad():
                prototype = forward_kinematics(
                    torch.tensor(source[None], device=args.device)
                )
            if task == "arm_circle":
                predicted_arm_vectors = (
                    predicted_positions[:, :, [20, 21]]
                    - predicted_positions[:, :, [16, 17]]
                )
                prototype_arm_vectors = (
                    prototype[:, :, [20, 21]] - prototype[:, :, [16, 17]]
                )
                loss = (
                    loss
                    + 5
                    * (
                        torch.nn.functional.normalize(predicted_arm_vectors, dim=-1)
                        - torch.nn.functional.normalize(prototype_arm_vectors, dim=-1)
                    )
                    .square()
                    .mean()
                )
            else:
                gap = (
                    predicted_positions[:, :, 20] - predicted_positions[:, :, 21]
                ).norm(dim=-1)
                prototype_hand_gap = (prototype[:, :, 20] - prototype[:, :, 21]).norm(
                    dim=-1
                )
                pattern = (
                    prototype_hand_gap - prototype_hand_gap.amin(1, keepdim=True)
                ) / (
                    prototype_hand_gap.amax(1, keepdim=True)
                    - prototype_hand_gap.amin(1, keepdim=True)
                ).clamp_min(
                    0.05
                )
                loss = loss + 25 * (gap - (0.12 + value * pattern)).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.denoiser.controller.parameters(), 1.0)
        optimizer.step()
        logs.append(
            dict(
                step=step,
                task=task,
                loss=float(loss),
                command=value,
                quantity=float(measured_quantity),
            )
        )
        if step % 50 == 0:
            print(logs[-1], flush=True)
    destination = Path(args.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    save(model, checkpoint, destination, args.steps)
    destination.with_suffix(".training.json").write_text(
        json.dumps(
            dict(
                initial=args.initial,
                manifest=args.manifest,
                sampling="Task-balanced cycles; all train records retained",
                seed=args.seed,
                steps=args.steps,
                log=logs,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
