"""Joint task-balanced training through the deployed DDIM sampler."""

import argparse, json, random
from pathlib import Path
import numpy as np, torch
from shared_motion.data import read_manifest, BalancedTaskSampler
from .model import load, save, command, TASKS
from .schema import RANGES, TASKS as OLD
from .catalog import NEW, quantities
from .kinematics import FK, quantity
from .sampling import generate


def differentiable(model, z, c, seed):
    local = z["local"][None]
    b, n, _ = local.shape
    device = local.device
    local = model.motion_normalizer(
        torch.cat([torch.zeros(b, n, 205, device=device), local], -1)
    )[..., 205:]
    y = dict(
        mask=torch.ones(b, n, device=device, dtype=torch.bool),
        tx=model.prepare_tx_emb(z["tx"]),
    )
    noise = torch.randn(
        b,
        n,
        205,
        generator=torch.Generator(device=device).manual_seed(seed),
        device=device,
    )
    steps = np.linspace(model.timesteps - 1, 0, 50, dtype=int)
    model.denoiser.static_residuals = model.denoiser.controller(c)
    try:
        for i, t in enumerate(steps):
            x = torch.cat([noise, local], -1)
            pred = model.denoiser(
                x, y, torch.full((b,), int(t), device=device, dtype=torch.long)
            )
            if i == len(steps) - 1:
                return model.motion_normalizer.inverse(pred)[..., :205]
            al = model.alphas_cumprod[t]
            an = model.alphas_cumprod[steps[i + 1]]
            noise = (
                an.sqrt() * pred
                + (1 - an).sqrt() * (x - al.sqrt() * pred) / (1 - al).sqrt()
            )[..., :205]
    finally:
        model.denoiser.static_residuals = None


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--initial", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--skeleton", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=2700)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--lr", type=float, default=3e-5)
    a = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    rows = read_manifest(a.manifest)
    if any(r["task"] not in TASKS or not r.get("prompt_cache") for r in rows):
        raise ValueError("Training rows need a supported task and prompt_cache")
    model, pack = load(a.initial, a.device)
    teacher, _ = load(a.initial, a.device)
    model.requires_grad_(False)
    model.denoiser.controller.requires_grad_(True)
    teacher.requires_grad_(False)
    fk = FK(a.skeleton, a.device)
    optimizer = torch.optim.AdamW(
        model.denoiser.controller.parameters(), lr=a.lr, weight_decay=1e-6
    )
    cache = {}
    logs = []
    for step, index in enumerate(BalancedTaskSampler(rows, a.steps, a.seed), 1):
        row = rows[index]
        task = row["task"]
        path = row["prompt_cache"]
        if path not in cache:
            cache[path] = torch.load(path, map_location=a.device, weights_only=False)
        z = cache[path]
        frames = len(z["local"])
        lo, hi = NEW[task]["bounds"] if task in NEW else RANGES[OLD.index(task)]
        value = float(row.get("command", rng.uniform(lo, hi)))
        seed = rng.randrange(2**31)
        c = command(task, value, frames, a.device)
        with torch.no_grad():
            target = generate(teacher, z["local"], z["tx"], c, seed)
            tp = fk(target)
        pred = differentiable(model, z, c, seed)
        pp = fk(pred)
        q = (
            quantities(pp)[task]
            if task in NEW
            else quantity(pp, task, scale=1.2701193988323212 / fk.height)
        )
        qloss = ((q - value) / (hi - lo)).square().mean()
        geom = ((pp - pp[:, :, :1]) - (tp - tp[:, :, :1])).square().mean() + 0.1 * (
            pp[:, :, 0] - tp[:, :, 0]
        ).square().mean()
        vel = (
            (pp[:, 1:] - pp[:, :-1]) - (tp[:, 1:] - tp[:, :-1])
        ).square().mean() * 400
        replay = []
        for t in TASKS:
            low, high = NEW[t]["bounds"] if t in NEW else RANGES[OLD.index(t)]
            replay.append(command(t, rng.uniform(low, high), frames, a.device))
        cc = torch.cat(replay, 0)
        with torch.no_grad():
            tr, to = teacher.denoiser.controller(cc)
        sr, so = model.denoiser.controller(cc)
        retain = (sr - tr).square().mean() + (so - to).square().mean()
        loss = qloss + 2 * geom + 0.05 * vel + 300 * retain
        # Preserve the selected stationary support prior for the nine added classes.
        if task in NEW and task not in ["jog", "march"]:
            feet = pp[:, :, [7, 8]]
            loss = (
                loss
                + 0.4 * ((pp[:, 1:, 0, :2] - pp[:, :-1, 0, :2]) * 20).square().mean()
                + 0.1 * ((feet[:, 1:] - feet[:, :-1]) * 20).square().mean()
            )
            if task != "squat":
                ankles = feet.mean(2)
                height = pp[:, :, 0, 2] - ankles[:, :, 2]
                offset = (pp[:, :, 0, :2] - ankles[:, :, :2]).norm(dim=-1)
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
                [np.interp(sample, np.arange(len(source)), v) for v in source.T], 1
            ).astype("float32")
            with torch.no_grad():
                prototype = fk(torch.tensor(source[None], device=a.device))
            if task == "arm_circle":
                v = pp[:, :, [20, 21]] - pp[:, :, [16, 17]]
                pv = prototype[:, :, [20, 21]] - prototype[:, :, [16, 17]]
                loss = (
                    loss
                    + 5
                    * (
                        torch.nn.functional.normalize(v, dim=-1)
                        - torch.nn.functional.normalize(pv, dim=-1)
                    )
                    .square()
                    .mean()
                )
            else:
                gap = (pp[:, :, 20] - pp[:, :, 21]).norm(dim=-1)
                pg = (prototype[:, :, 20] - prototype[:, :, 21]).norm(dim=-1)
                pattern = (pg - pg.amin(1, keepdim=True)) / (
                    pg.amax(1, keepdim=True) - pg.amin(1, keepdim=True)
                ).clamp_min(0.05)
                loss = loss + 25 * (gap - (0.12 + value * pattern)).square().mean()
        if not torch.isfinite(loss):
            raise RuntimeError("Non-finite training loss")
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.denoiser.controller.parameters(), 1.0)
        optimizer.step()
        logs.append(
            dict(
                step=step, task=task, loss=float(loss), command=value, quantity=float(q)
            )
        )
        if step % 50 == 0:
            print(logs[-1], flush=True)
    dest = Path(a.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    save(model, pack, dest, a.steps)
    dest.with_suffix(".training.json").write_text(
        json.dumps(
            dict(
                initial=a.initial,
                manifest=a.manifest,
                sampling="Task-balanced cycles; all train records retained",
                seed=a.seed,
                steps=a.steps,
                log=logs,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
