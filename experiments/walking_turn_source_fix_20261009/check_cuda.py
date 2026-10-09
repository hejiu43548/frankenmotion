"""Official-weight integration check; no saved model or long training run."""

import json
from pathlib import Path

from hydra import compose, initialize_config_dir
from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch

from shared_motion.training.catalog import TASK_NAMES
from shared_motion.training.data import MotionDataset
from shared_motion.training.geometry import Skeleton
from shared_motion.training.model import build_model
from shared_motion.training.runner import frozen_state, verify_frozen
from shared_motion.training.turn import TURN_NATIVE_SPEED


def main():
    repository = Path.cwd()
    destination = Path(
        "/mnt/sda2/frankenmotion/outputs_amass/main_walking_turn_validation_20261009"
    )
    destination.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    torch.backends.mha.set_fastpath_enabled(False)
    data = MotionDataset(
        "/mnt/sda2/frankenmotion/outputs_amass/main_walking_turn_data_20261009/train.json",
        "train",
        ["turn"],
        "/home/psirobot/projects/frankenmotion",
    )
    indices = [
        next(
            index
            for index, row in enumerate(data.rows)
            if row["direction"] == direction
        )
        for direction in ["left", "right"]
    ]
    batch = data.batch(indices, "cuda")
    skeleton_path = "/home/psirobot/projects/frankenmotion/outputs_amass/transfer_charlie_20261008/snapshot/outputs_amass/franken_eleven_20261003/skeleton.npz"
    skeleton = Skeleton(skeleton_path).cuda()
    evidence = {}
    for controller in ["with_root", "without_root"]:
        with initialize_config_dir(
            config_dir=str(repository / "config"), version_base="1.3"
        ):
            config = compose(
                config_name="train",
                overrides=[
                    "stage=stage2",
                    f"controller={controller}",
                    "backbone.checkpoint=/home/psirobot/checkpoints/frankenmotion/frankenmotion.ckpt",
                ],
            )
        model = build_model(config, "task").cuda()
        snapshot = frozen_state(model)
        optimizer = torch.optim.AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=1e-4,
        )
        model.train()
        loss, _ = instantiate(config.loss).supervised(model, skeleton, batch)
        assert torch.isfinite(loss)
        loss.backward()
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        verify_frozen(model, snapshot)
        reference = build_model(config, "task").cuda()
        reference.load_adapter(model.adapter_state())
        reference.requires_grad_(False).eval()
        with initialize_config_dir(
            config_dir=str(repository / "config"), version_base="1.3"
        ):
            free_config = compose(config_name="train", overrides=["stage=stage3"])
        commands = torch.tensor([-1.2, 1.2], device="cuda")
        control = model.requested_root(batch, commands, skeleton)
        assert torch.allclose(
            control[:, 0, 0] * 3, torch.full((2,), TURN_NATIVE_SPEED, device="cuda")
        )
        free_loss, _ = instantiate(free_config.loss).free(
            model, reference, skeleton, batch, commands, [801, 802]
        )
        assert torch.isfinite(free_loss)
        free_loss.backward()
        assert all(
            parameter.grad is not None and torch.isfinite(parameter.grad).all()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
        verify_frozen(model, snapshot)
        model.eval()
        with torch.no_grad():
            generated = model.sample(batch, commands, skeleton, [803, 804], 50)
            poisoned = dict(
                batch,
                motion=torch.full_like(batch["motion"], float("nan")),
                quantity=torch.full_like(batch["quantity"], float("nan")),
            )
            repeated = model.sample(poisoned, commands, skeleton, [803, 804], 50)
            assert torch.equal(generated, repeated)
        evidence[controller] = dict(
            backbone_sha256=model.backbone_sha256,
            task_policy=model.task_policy,
            supervised_loss=float(loss),
            free_loss=float(free_loss),
            stage2_and_stage3_finite_gradients=True,
            frozen_parameters_unchanged=True,
            requested_native_speed_m_s=TURN_NATIVE_SPEED,
            generation_ignores_gt_motion_and_quantity=True,
            source_keys=[data.rows[index]["key"] for index in indices],
            config=OmegaConf.to_container(config.controller),
        )
        del loss, free_loss, generated, repeated, model, reference, optimizer, snapshot
        torch.cuda.empty_cache()
    (destination / "cuda_check.json").write_text(json.dumps(evidence, indent=2) + "\n")
    print(json.dumps(evidence, indent=2))


if __name__ == "__main__":
    main()
