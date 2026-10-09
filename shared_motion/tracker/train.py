"""Task-balanced fine tuning of the ONE exported shared correction head."""

import argparse, copy, json, random
from pathlib import Path
import numpy as np, torch
from shared_motion.data import read_manifest, BalancedTaskSampler


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--initial", required=True)
    p.add_argument("--manifest", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--steps", type=int, default=4000)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--lr", type=float, default=3e-5)
    p.add_argument("--retain", type=float, default=1.0)
    a = p.parse_args()
    torch.set_num_threads(2)
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)
    rows = read_manifest(a.manifest)
    net = torch.jit.load(a.initial, map_location=a.device).eval()
    teacher = torch.jit.load(a.initial, map_location=a.device).eval()
    for parameter in net.parameters():
        parameter.requires_grad_(False)
    for parameter in net.head.parameters():
        parameter.requires_grad_(True)
    for parameter in teacher.parameters():
        parameter.requires_grad_(False)
    opt = torch.optim.Adam(net.head.parameters(), lr=a.lr)
    cache = {}
    sampler = BalancedTaskSampler(rows, a.steps * a.batch_size, a.seed)
    it = iter(sampler)

    def sample(row):
        path = row["path"]
        if path not in cache:
            z = np.load(path)
            x = np.c_[z["observations"], z["sonic_actions"], z["latents"]]
            y = z["actions"]
            n = len(y) if row.get("complete", True) else max(0, len(y) - 75)
            if n == 0:
                raise ValueError(f"No usable frames: {path}")
            if x.shape != (len(y), 588) or y.shape[1:] != (29,):
                raise ValueError(f"Invalid teacher dataset: {path}")
            cache[path] = (x[:n].astype("float32"), y[:n].astype("float32"))
        x, y = cache[path]
        i = rng.randrange(len(y))
        return x[i], y[i]

    for step in range(a.steps):
        batch = [sample(rows[next(it)]) for _ in range(a.batch_size)]
        x = torch.as_tensor(np.stack([r[0] for r in batch]), device=a.device)
        y = torch.as_tensor(np.stack([r[1] for r in batch]), device=a.device)
        normalized = torch.clamp((x - net.mean) / net.std, -10, 10)
        pred = x[:, 495:524] + 3 * torch.tanh(net.head(normalized))
        with torch.no_grad():
            old = x[:, 495:524] + 3 * torch.tanh(
                teacher.head(torch.clamp((x - teacher.mean) / teacher.std, -10, 10))
            )
        loss = (pred - y).square().mean() + a.retain * (pred - old).square().mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.head.parameters(), 2.0)
        opt.step()
        if step % 100 == 0:
            print(step, float(loss), flush=True)
    net.reset()
    dest = Path(a.output)
    dest.parent.mkdir(parents=True, exist_ok=True)
    torch.jit.save(net.cpu(), str(dest))
    side = json.loads(Path(a.initial).with_suffix(".json").read_text())
    side.update(
        training_manifest=str(Path(a.manifest).resolve()),
        sampling="Uniform task cycles; uniform clip then frame; no corpus truncation",
        steps=a.steps,
    )
    dest.with_suffix(".json").write_text(json.dumps(side, indent=2))


if __name__ == "__main__":
    main()
