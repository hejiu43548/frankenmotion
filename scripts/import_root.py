#!/usr/bin/env python3
"""Import a task-agnostic root adapter with explicit, immutable source lineage."""

import importlib.util
from pathlib import Path
import sys

import hydra
from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared_motion.training.model import build_model, file_sha256
from shared_motion.training.runner import save_json


@hydra.main(version_base="1.3", config_path="../config", config_name="import_root")
def main(config):
    output = Path(config.output)
    if output.exists():
        raise FileExistsError(output)
    if file_sha256(config.source_checkpoint) != config.expected_sha256:
        raise ValueError("Root checkpoint SHA256 mismatch")
    source = torch.load(
        config.source_checkpoint, map_location="cpu", weights_only=False
    )
    expected_code_hash = source["protocol"]["source_sha256"][
        "work/charlie_aligned/root_control.py"
    ]
    if file_sha256(config.reference_code) != expected_code_hash:
        raise ValueError("Original root implementation hash mismatch")
    if source["protocol"]["official_sha256"] != config.backbone.expected_sha256:
        raise ValueError("Root source was trained against another official backbone")
    model = build_model(config, "root").to(config.device).eval()
    model.load_adapter(source["adapter"])
    specification = importlib.util.spec_from_file_location(
        "verified_source_root", config.reference_code
    )
    module = importlib.util.module_from_spec(specification)
    specification.loader.exec_module(module)
    bundle = instantiate(config.backbone)
    reference = module.RootControl(bundle["denoiser"]).to(config.device).eval()
    reference.load_adapter(source["adapter"])
    assert all(
        torch.equal(value, source["adapter"][name])
        for name, value in model.adapter_state().items()
    )
    torch.manual_seed(20261009)
    motion = torch.randn(4, 40, 613, device=config.device)
    control = torch.randn(4, 40, 4, device=config.device)
    control[..., 2:] = torch.tensor(
        [[0, 0], [1, 0], [0, 1], [1, 1]], device=config.device
    )[:, None]
    conditioning = dict(
        mask=torch.ones(4, 40, dtype=torch.bool, device=config.device),
        tx=dict(
            x=torch.randn(4, 1, 512, device=config.device),
            mask=torch.ones(4, 1, dtype=torch.bool, device=config.device),
        ),
        root_control=control,
    )
    differences = []
    with torch.no_grad():
        for timestep in [0, 49, 99]:
            timesteps = torch.full(
                (4,), timestep, device=config.device, dtype=torch.long
            )
            expected = reference(motion, conditioning, timesteps)
            actual = model.denoiser(motion, conditioning, timesteps)
            differences.append(float((actual - expected).abs().max()))
            if not torch.equal(actual, expected):
                raise ValueError(
                    "Root import forward output differs from source implementation"
                )
    lineage = dict(
        checkpoint=str(Path(config.source_checkpoint).resolve()),
        checkpoint_sha256=config.expected_sha256,
        reference_code_sha256=expected_code_hash,
        epoch=source["epoch"],
        optimizer_step=source["step"],
        training_protocol=source["protocol"],
        scope="Task-agnostic speed/yaw profile adapter trained on previous11-class-derived root data; not retrained on20 classes or walking_turn_v2",
    )
    checks = dict(
        parameters_exact=True,
        forward_bitwise_equal=True,
        forward_max_errors=differences,
        root_trainable_parameters=sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        ),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        dict(
            format="verified_external_root_v1",
            controller_kind="charlie_root",
            backbone_sha256=model.backbone_sha256,
            backbone_config=model.backbone_configuration,
            adapter=model.adapter_state(),
            source=lineage,
            import_checks=checks,
        ),
        output,
    )
    save_json(
        output.with_suffix(".json"),
        dict(source=lineage, checks=checks, imported_sha256=file_sha256(output)),
    )
    print(checks, flush=True)


if __name__ == "__main__":
    main()
