"""Walking-turn semantics shared by supervision, generation and fixed scans."""

import hashlib

import torch

TURN_REVISION = "walking_turn_v2"
TURN_RANGE = (-3.5, 3.5)
TURN_MIN_MAGNITUDE = 0.35
TURN_NATIVE_SPEED = 0.7448640465736389
TURN_POLICY = dict(
    revision=TURN_REVISION,
    task="walking while turning; translation required",
    angle="signed unwrapped radians; positive right, negative left",
    range_rad=list(TURN_RANGE),
    native_speed_m_s=TURN_NATIVE_SPEED,
    speed_source="median over591 admitted TRAIN event crops only",
    training_magnitude_range_rad=[TURN_MIN_MAGNITUDE, TURN_RANGE[1]],
    training_direction="match source turn direction",
    spin_added=False,
)


def signed_turn(positions):
    side = positions[:, :, 1, :2] - positions[:, :, 2, :2]
    heading = torch.atan2(side[..., 1], side[..., 0])
    increments = heading[:, 1:] - heading[:, :-1]
    return -torch.atan2(increments.sin(), increments.cos()).sum(1)


def require_turn_revision(record):
    if record.get("turn_source_revision") != TURN_REVISION:
        raise ValueError(
            "Unverified/legacy turn source: run scripts/repair_turn_data.py to use walking_turn_v2 event crops"
        )


def validate_turn_record(record, cache_hash, frames, quantity):
    require_turn_revision(record)
    if record.get("cache_sha256") != cache_hash:
        raise ValueError("Walking-turn cache hash does not match its verified source")
    if (
        not record.get("ready_for_training")
        or record.get("pad_frames") != 0
        or record.get("real_frames") != frames
        or record.get("crop_end_frame_20fps", 0)
        - record.get("crop_start_frame_20fps", 0)
        != frames
        or not 40 <= frames <= 120
        or record.get("admission_reasons") != []
        or record.get("direction") not in ["left", "right"]
        or not abs(quantity) > 0
        or (quantity > 0) != (record["direction"] == "right")
    ):
        raise ValueError("Invalid walking-turn event crop, direction or admission")


def scan_indices(dataset, task_index, commands, turn_index):
    indices = dataset.groups[task_index]

    def identity(index):
        return hashlib.sha256(str(dataset.rows[index]["key"]).encode()).hexdigest()

    if task_index != turn_index:
        return [min(indices, key=identity)] * len(commands)
    selected = {}
    for direction in ["left", "right"]:
        opposite = "right" if direction == "left" else "left"
        pool = [
            index for index in indices if dataset.rows[index]["direction"] == direction
        ]
        if not pool:
            raise ValueError(
                "Walking-turn validation requires both left and right sources"
            )
        selected[direction] = min(
            pool,
            key=lambda index: (
                opposite in dataset.rows[index].get("caption", "").lower(),
                direction not in dataset.rows[index].get("caption", "").lower(),
                dataset.rows[index]["real_frames"] < 80,
                identity(index),
            ),
        )
    return [selected["left" if float(value) < 0 else "right"] for value in commands]


def sample_commands(batch, command_ranges, turn_index):
    from .reach import REACH_INDEX, command_bounds, command_vectors, dimension_mask

    quantities = command_vectors(batch["quantity"], batch["task"])
    bounds = command_bounds(quantities, batch["task"], command_ranges)
    commands = bounds[..., 0] + torch.rand_like(quantities) * (
        bounds[..., 1] - bounds[..., 0]
    )
    commands *= dimension_mask(batch["task"])
    turning = batch["task"] == turn_index
    magnitudes = TURN_MIN_MAGNITUDE + torch.rand(
        int(turning.sum()), device=commands.device
    ) * (TURN_RANGE[1] - TURN_MIN_MAGNITUDE)
    commands[turning, 0] = magnitudes * quantities[turning, 0].sign()
    # Real target support only; a Cartesian bounding box is not a reachable workspace.
    reaching = batch["task"] == REACH_INDEX
    commands[reaching] = quantities[reaching]
    return commands
