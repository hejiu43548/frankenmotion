import argparse
import numpy as np
import torch
from .model import load, command
from .sampling import generate


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--prompt-cache", required=True)
    parser.add_argument("--task", required=True)
    parser.add_argument("--command", type=float, required=True)
    parser.add_argument("--extra", type=float, nargs="+")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    torch.set_num_threads(2)
    model, _ = load(args.checkpoint, args.device)
    prompt_cache = torch.load(
        args.prompt_cache, map_location=args.device, weights_only=False
    )
    command_features = command(
        args.task, args.command, len(prompt_cache["local"]), args.device, args.extra
    )
    motion = generate(
        model, prompt_cache["local"], prompt_cache["tx"], command_features, args.seed
    )
    np.savez_compressed(
        args.output,
        motion=motion[0].cpu().numpy(),
        control_features=command_features[0].cpu().numpy(),
        fps=20.0,
        task=args.task,
        command=args.command,
        seed=args.seed,
    )


if __name__ == "__main__":
    main()
