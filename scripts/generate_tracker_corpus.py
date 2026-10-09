"""Generate seed-disjoint official motion references; no policy is loaded."""

import hashlib
import json
import os
from pathlib import Path
import sys

import hydra
import numpy as np
from hydra.utils import instantiate
from omegaconf import DictConfig
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.config import read_config
from src.data.text_part_utils import load_from_annotation_with_model
from src.tools.inference import load_diffusion
from src.tools.inference import load_smplh
from src.tools.parse_user_input import parse_and_validate_user_input
from src.tools.extract_joints import extract_joints


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def prepare_batch(annotation, encoder, frames, device):
    embeddings, texts = load_from_annotation_with_model(
        text_encoder=encoder,
        annotations=annotation["annotations"],
        path=annotation["path"],
        start=annotation["start"],
        end=annotation["end"],
    )
    local = embeddings["local"]["x"]
    local_mask = embeddings["local"]["mask"]
    if local.shape[0] != frames or local.shape[1] != 408:
        raise ValueError(f"Unexpected text feature shape {local.shape}")
    motion = torch.zeros(frames, 205)
    batch = {
        "x": torch.cat([motion, local], dim=1)[None].to(device),
        "motion_dim": 205,
        "text_dim": 408,
        "text": [texts],
        "tx": {
            "local": {
                "x": local[None].to(device),
                "mask": local_mask[None].to(device),
                "length": embeddings["local"]["length"],
            },
            "x": embeddings["x"][None].to(device),
            "length": torch.tensor([embeddings["length"]], device=device),
        },
        "length": torch.tensor([frames]),
        "mask": torch.ones(1, frames, dtype=torch.bool),
        "keyid": ["generated"],
        "global_text": [annotation["sequence_caption"]],
        "inpainting_mask": torch.cat(
            [torch.ones_like(motion), torch.zeros_like(local)], dim=1
        ).to(device),
        "stats_mask": torch.cat([torch.ones_like(motion), local_mask], dim=1)[None].to(
            device
        ),
    }
    unconditional = encoder("unknown")
    batch["tx_uncond_batch"] = {
        "x": unconditional["x"][None].to(device),
        "length": torch.tensor([unconditional["length"]], device=device),
    }
    return batch, texts


@hydra.main(
    config_path="../config/tracker_rl", config_name="corpus", version_base="1.3"
)
def main(configuration: DictConfig):
    torch.set_num_threads(2)
    os.chdir(configuration.repository)
    output = Path(configuration.output)
    output.mkdir(parents=True, exist_ok=True)
    checkpoint_directory = Path(configuration.checkpoint_directory)
    if (
        digest_file(checkpoint_directory / "frankenmotion.ckpt")
        != configuration.checkpoint_sha256
    ):
        raise ValueError("Official checkpoint hash mismatch")
    model_configuration = read_config(str(checkpoint_directory))
    encoder = instantiate(model_configuration.data.text_encoder)
    encoder.no_model = False
    encoder.rand_mask = False
    model = load_diffusion(
        model_configuration,
        str(checkpoint_directory),
        "frankenmotion",
        configuration.device,
    )
    smpl = load_smplh(gender="male")
    records = []
    for prompt_path in sorted(Path(configuration.prompts).glob("*.json")):
        annotation = parse_and_validate_user_input(
            str(prompt_path), model_configuration, configuration.fps
        )
        batch, texts = prepare_batch(
            annotation, encoder, configuration.frames, configuration.device
        )
        infos = {
            "all_texts": [texts],
            "all_lengths": [configuration.frames],
            "output_lengths": [configuration.frames],
            "global_texts": [annotation["sequence_caption"]],
            "featsname": model_configuration.motion_features,
            "guidance_weight": 1.0,
        }
        for split in ["train", "val", "test"]:
            for seed in configuration[split + "_seeds"]:
                destination = output / f"{prompt_path.stem}_{seed}.npz"
                if destination.exists():
                    raise FileExistsError(destination)
                torch.manual_seed(seed)
                np.random.seed(seed)
                with torch.inference_mode():
                    features = model.batch_forward(batch, infos).cpu()[0]
                    decoded = extract_joints(
                        features[:, :205],
                        "smplrifke",
                        fps=20,
                        value_from="smpl",
                        smpl_layer=smpl,
                    )
                np.savez_compressed(
                    destination,
                    motion=features[:, :205].numpy(),
                    poses=decoded["smpldata"]["poses"],
                    trans=decoded["smpldata"]["trans"],
                    joints=decoded["joints"][:, :22],
                    fps=20,
                )
                records.append(
                    {
                        "task": prompt_path.stem,
                        "seed": seed,
                        "split": split,
                        "path": str(destination),
                        "sha256": digest_file(destination),
                        "prompt": annotation["sequence_caption"],
                    }
                )
                temporary = output / "manifest.tmp"
                temporary.write_text(json.dumps(records, indent=2))
                temporary.replace(output / "manifest.json")
                print(
                    f"GENERATED {len(records)}/160 {prompt_path.stem} {seed}",
                    flush=True,
                )
    (output / "complete.json").write_text(
        json.dumps(
            {
                "count": len(records),
                "checkpoint_sha256": configuration.checkpoint_sha256,
            }
        )
    )


if __name__ == "__main__":
    main()
