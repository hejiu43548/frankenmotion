import json
from pathlib import Path
import sys
import numpy as np
import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf

sys.path.insert(0, "/home/pku/frankenmotion/work/fixed_target_20261010/repository")
from shared_motion.training.reach3d_model import ReachDiffusion
from shared_motion.training.geometry import Skeleton
from shared_motion.training.fixed_target import (
    FixedTargetPolicy,
    fixed_observations,
    apply_fixed_residual,
    fixed_metrics,
)
from shared_motion.training.model import file_sha256

root = Path("/home/pku/frankenmotion/work/fixed_target_20261010")
out = root / "position_sweep"
out.mkdir(exist_ok=True)
torch.set_num_threads(4)
checkpoint = torch.load(
    root / "reach_sft_seed84001/final.pt", map_location="cpu", weights_only=False
)
policy_checkpoint = torch.load(
    root / "reach_ppo_seed86001/final.pt", map_location="cpu", weights_only=False
)
assert policy_checkpoint["initial_sha256"] == file_sha256(
    root / "reach_sft_seed84001/final.pt"
)
training = OmegaConf.create(checkpoint["config"])
audit = checkpoint["audit"]
model = (
    ReachDiffusion(
        instantiate(training.backbone),
        audit["train_target_center"],
        audit["train_target_scale"],
    )
    .cuda()
    .eval()
)
model.load_adapter(checkpoint["adapter"])
policy = FixedTargetPolicy().cuda().eval()
policy.load_state_dict(policy_checkpoint["policy"])
skeleton = Skeleton(training.skeleton).cuda()
archive = np.load(root / "reach_data/text_template.npz")
requests = []
for axis, values in [
    ("forward", [0.25, 0.35, 0.45, 0.55, 0.65]),
    ("lateral", [-0.5, -0.35, -0.2, -0.05, 0.10]),
    ("height", [1.1, 1.3, 1.5, 1.7, 1.85]),
]:
    for index, value in enumerate(values):
        target = [0.45, -0.2, 1.5]
        target[["forward", "lateral", "height"].index(axis)] = value
        requests.append(dict(key=f"{axis}_{index}", axis=axis, target=target))
count = len(requests)
frames = len(archive["local"])
text = torch.tensor(archive["tx"]).float().cuda()[None].repeat(count, 1)
local = torch.tensor(archive["local"]).float().cuda()[None].repeat(count, 1, 1)
local_mask = torch.tensor(archive["local_mask"]).bool().cuda()[None].repeat(count, 1, 1)
mask = torch.ones(count, frames, device="cuda", dtype=torch.bool)
hands = torch.ones(count, device="cuda", dtype=torch.long)
events = torch.full(
    (count,), int(archive["event_frames"]), device="cuda", dtype=torch.long
)
target = torch.tensor([row["target"] for row in requests], device="cuda")
with torch.no_grad():
    base = model.sample(
        text, local, local_mask, mask, hands, target, [51000] * count, steps=20
    )
    state = fixed_observations(base, target, events, hands, skeleton)
    motion = apply_fixed_residual(
        base,
        policy.mean(state),
        events,
        hands,
        skeleton,
        policy_checkpoint["config"]["max_angle"],
        policy_checkpoint["config"]["radius_frames"],
    )
    joints = skeleton(motion)
    metrics = fixed_metrics(joints, target, hands, events, "reach")
    measured = joints[:, -5:, 21].mean(1)
for index, row in enumerate(requests):
    row.update(
        measured=measured[index].tolist(),
        error_cm=float(metrics["error_m"][index]) * 100,
        hold_speed_m_s=float(metrics["final_hold_speed_m_s"][index]),
        seed=51000,
        frames=frames,
    )
    np.savez(
        out / (row["key"] + ".npz"),
        motion=motion[index].cpu().numpy(),
        joints=joints[index].cpu().numpy(),
        target=target[index].cpu().numpy(),
    )
report = dict(
    protocol="Same checkpoint, right hand, text template, duration, initial noise seed 51000. Only fixed WORLD target XYZ changes. Unfiltered preset axis sweeps; includes sparse-support/extrapolation cases. Z is height above ground. No retraining or best-seed selection.",
    generator_sha256=file_sha256(root / "reach_sft_seed84001/final.pt"),
    policy_sha256=file_sha256(root / "reach_ppo_seed86001/final.pt"),
    text_template_sha256=file_sha256(root / "reach_data/text_template.npz"),
    rows=requests,
)
(out / "commands.json").write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps(requests, indent=2))
