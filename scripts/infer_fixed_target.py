"""Generate reach/strike toward a point fixed before generation.

The text-template artifact contains no reference motion. The RL residual observes
only the newly generated motion. Exported 205-D rotations and joint features agree.
"""

import json
from pathlib import Path
import sys

import hydra
from hydra.utils import instantiate
import numpy as np
from omegaconf import OmegaConf
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from shared_motion.training.geometry import Skeleton
from shared_motion.training.model import file_sha256
from shared_motion.training.reach3d_model import ReachDiffusion
from shared_motion.training.fixed_target import (
    FixedTargetPolicy,
    fixed_observations,
    apply_fixed_residual,
    fixed_metrics,
)
from src.tools.geometry import axis_angle_rotation


@hydra.main(
    version_base="1.3", config_path="../config", config_name="infer_fixed_target"
)
def main(config):
    torch.set_num_threads(4)
    checkpoint = torch.load(config.generator, map_location="cpu", weights_only=False)
    policy_checkpoint = torch.load(
        config.policy, map_location="cpu", weights_only=False
    )
    if (
        checkpoint.get("format") != "fixed_world_adapter_v1"
        or policy_checkpoint.get("format") != "fixed_world_arm_ppo_v1"
    ):
        raise ValueError("Expected fixed-world checkpoints")
    if policy_checkpoint["initial_sha256"] != file_sha256(config.generator):
        raise ValueError("The arm policy requires its exact trained generator")
    training = OmegaConf.create(checkpoint["config"])
    training.backbone.checkpoint = config.backbone_checkpoint
    audit = checkpoint["audit"]
    model = (
        ReachDiffusion(
            instantiate(training.backbone),
            audit["train_target_center"],
            audit["train_target_scale"],
        )
        .to(config.device)
        .eval()
    )
    if model.backbone_sha256 != checkpoint["backbone_sha256"]:
        raise ValueError("Official backbone fingerprint mismatch")
    model.load_adapter(checkpoint["adapter"])
    policy = FixedTargetPolicy().to(config.device).eval()
    policy.load_state_dict(policy_checkpoint["policy"], strict=True)
    skeleton = Skeleton(config.skeleton).to(config.device)
    with np.load(config.text_template, allow_pickle=False) as archive:
        if set(archive.files) != {"tx", "local", "local_mask", "event_frames"}:
            raise ValueError(
                "Expected a text/timing-only template, without source motion"
            )
        text = torch.tensor(archive["tx"], device=config.device, dtype=torch.float32)[
            None
        ]
        local = torch.tensor(
            archive["local"], device=config.device, dtype=torch.float32
        )[None]
        local_mask = torch.tensor(
            archive["local_mask"], device=config.device, dtype=torch.bool
        )[None]
        event_frames = torch.tensor(
            [int(archive["event_frames"])], device=config.device
        )
    target = torch.tensor(
        [list(config.target)], device=config.device, dtype=torch.float32
    )
    if target.shape != (1, 3) or not torch.isfinite(target).all():
        raise ValueError("Specify finite XYZ meters")
    if config.target_frame == "initial_body":
        # External initial anchor, resolved once BEFORE sampling, never from its output.
        origin = torch.tensor(
            [list(config.initial_pelvis_world)],
            device=config.device,
            dtype=torch.float32,
        )
        yaw = torch.tensor(float(config.initial_yaw_radians), device=config.device)
        target = origin + (axis_angle_rotation("Z", yaw) @ target[..., None]).squeeze(
            -1
        )
    elif config.target_frame != "world":
        raise ValueError("target_frame must be world or initial_body")
    mask = torch.ones(local.shape[:2], device=config.device, dtype=torch.bool)
    if config.hand not in ("left", "right"):
        raise ValueError("hand must be left or right")
    hand = 0 if config.hand == "left" else 1
    if hand not in audit["hands"]:
        raise ValueError("Requested hand has no audited training data")
    hands = torch.tensor([hand], device=config.device, dtype=torch.long)
    with torch.no_grad():
        base = model.sample(
            text,
            local,
            local_mask,
            mask,
            hands,
            target,
            [config.seed],
            steps=config.steps,
        )
        state = fixed_observations(base, target, event_frames, hands, skeleton)
        corrected = apply_fixed_residual(
            base,
            policy.mean(state),
            event_frames,
            hands,
            skeleton,
            policy_checkpoint["config"]["max_angle"],
            policy_checkpoint["config"]["radius_frames"],
        )
        joints = skeleton(corrected)
        wrist = joints[0, event_frames[0], 20 + hand]
        metrics = fixed_metrics(joints, target, hands, event_frames, audit["task"])
    output = Path(config.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output,
        motion=corrected.cpu().numpy(),
        joints=joints.cpu().numpy(),
        target=target.cpu().numpy(),
        target_world=target.cpu().numpy(),
        hands=hands.cpu().numpy(),
        event_frames=event_frames.cpu().numpy(),
        seeds=np.array([config.seed]),
    )
    report = dict(
        command=list(config.target),
        command_frame=config.target_frame,
        target_world=target[0].tolist(),
        target_is_fixed=True,
        metrics={name: value[0].item() for name, value in metrics.items()},
        measured=wrist.tolist(),
        error_m=float((wrist - target[0]).norm()),
        generator_sha256=file_sha256(config.generator),
        policy_sha256=file_sha256(config.policy),
        official_sha256=model.backbone_sha256,
        template_sha256=file_sha256(config.text_template),
        reference_motion_loaded=False,
    )
    output.with_suffix(".json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
