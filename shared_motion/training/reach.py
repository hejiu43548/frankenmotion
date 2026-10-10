"""Versioned right-wrist XYZ target in the initial pelvis/body coordinate frame."""

import torch

REACH_REVISION = "right_wrist_xyz_v1"
REACH_INDEX = 1
RETIRED_POINT_INDEX = 14
REACH_BOUNDS = ((-0.15, 0.85), (-0.8, 0.8), (-0.6, 1.1))
REACH_HOLD_FRAMES = 5
REACH_POLICY = dict(
    revision=REACH_REVISION,
    merged_task="point",
    task="reach",
    target="mean right wrist position over final five valid frames",
    frame="initial pelvis origin; X initial forward, Y initial left, Z up; metres",
    command_dimensions=3,
    retired_task_id=RETIRED_POINT_INDEX,
    reserved_task_ids=True,
    bounds_m=[list(bounds) for bounds in REACH_BOUNDS],
)


def wrist_coordinates(positions):
    origin = positions[:, :1, 0]
    lateral = positions[:, 0, 1, :2] - positions[:, 0, 2, :2]
    lateral = lateral / lateral.norm(dim=-1, keepdim=True).clamp_min(1e-8)
    forward = torch.stack([lateral[:, 1], -lateral[:, 0]], dim=-1)
    relative = positions[:, :, 21] - origin
    return torch.stack(
        [
            (relative[..., :2] * forward[:, None]).sum(-1),
            (relative[..., :2] * lateral[:, None]).sum(-1),
            relative[..., 2],
        ],
        dim=-1,
    )


def target_xyz(positions):
    return wrist_coordinates(positions)[:, -REACH_HOLD_FRAMES:].mean(1)


def command_vectors(commands, task_indices):
    if (task_indices == RETIRED_POINT_INDEX).any():
        raise ValueError("point is merged into reach; use reach with an XYZ command")
    if commands.ndim == 1:
        if (task_indices == REACH_INDEX).any():
            raise ValueError("reach requires XYZ; scalar reach commands are obsolete")
        commands = torch.stack(
            [commands, torch.zeros_like(commands), torch.zeros_like(commands)], -1
        )
    if commands.shape != (len(task_indices), 3) or not torch.isfinite(commands).all():
        raise ValueError("Commands must be finite [batch,3] vectors")
    return commands


def dimension_mask(task_indices):
    return torch.stack(
        [
            torch.ones_like(task_indices, dtype=torch.bool),
            task_indices == REACH_INDEX,
            task_indices == REACH_INDEX,
        ],
        -1,
    )


def command_bounds(commands, task_indices, scalar_ranges):
    ranges = commands.new_tensor(scalar_ranges)[task_indices]
    bounds = commands.new_zeros(len(task_indices), 3, 2)
    bounds[:, :, 1] = 1
    bounds[:, 0] = ranges
    bounds[task_indices == REACH_INDEX] = commands.new_tensor(REACH_BOUNDS)
    return bounds


def normalized_commands(commands, task_indices, scalar_ranges):
    commands = command_vectors(commands, task_indices)
    bounds = command_bounds(commands, task_indices, scalar_ranges)
    normalized = (
        2 * (commands - bounds[..., 0]) / (bounds[..., 1] - bounds[..., 0]) - 1
    ).clamp(-5, 5)
    return normalized * dimension_mask(task_indices)


def validate_reach_record(record, cache_hash, frames, target):
    if record.get("semantic_revision") != REACH_REVISION:
        raise ValueError(
            "Legacy scalar reach data rejected; prepare right_wrist_xyz_v1"
        )
    if record.get("cache_sha256") != cache_hash or target.shape != (3,):
        raise ValueError("Invalid reach XYZ cache or source hash")
    if (
        record.get("pad_frames") != 0
        or record.get("real_frames") != frames
        or frames < 15
    ):
        raise ValueError("Reach requires unpadded actual approach and target hold")
    expected = target.new_tensor(record["target_xyz_m"])
    if not torch.allclose(target, expected, atol=1e-6):
        raise ValueError("Reach target differs from manifest")
