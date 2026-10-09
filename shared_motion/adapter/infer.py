import argparse, numpy as np, torch
from .model import load, command
from .sampling import generate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--prompt-cache", required=True)
    p.add_argument("--task", required=True)
    p.add_argument("--command", type=float, required=True)
    p.add_argument("--extra", type=float, nargs="+")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    a = p.parse_args()
    torch.set_num_threads(2)
    m, _ = load(a.checkpoint, a.device)
    z = torch.load(a.prompt_cache, map_location=a.device, weights_only=False)
    c = command(a.task, a.command, len(z["local"]), a.device, a.extra)
    raw = generate(m, z["local"], z["tx"], c, a.seed)
    np.savez_compressed(
        a.output,
        motion=raw[0].cpu().numpy(),
        control_features=c[0].cpu().numpy(),
        fps=20.0,
        task=a.task,
        command=a.command,
        seed=a.seed,
    )


if __name__ == "__main__":
    main()
