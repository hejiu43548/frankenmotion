"""Current staged task semantics; historical release schemas remain separate."""

import torch

from shared_motion.adapter.catalog import NEW
from shared_motion.adapter.schema import RANGES, TASKS
from shared_motion.adapter.kinematics import quantity
from .turn import TURN_RANGE, signed_turn

TASK_NAMES = TASKS + list(NEW)
# Stable checkpoint IDs; retired point slot is reserved, never sampled.
ACTIVE_TASK_NAMES = [name for name in TASK_NAMES if name != "point"]
ADDITIONAL_TASK_DEFINITIONS = {
    name: dict(definition) for name, definition in NEW.items()
}
ADDITIONAL_TASK_DEFINITIONS["march"].update(
    pattern=None,
    admission="run/jog act_cat plus stationary running gait checks",
    prompt="A person runs in place with a steady alternating running gait and stays at the same location.",
    unit="m mean peak ankle lift above per-foot support baseline",
    semantic_revision="stationary_running_v1",
)
ADDITIONAL_TASK_DEFINITIONS["jog"].update(
    pattern=None,
    admission="run/jog act_cat plus straight path, stable heading and continuous forward running gait checks",
    prompt="A person runs forward along a straight line without turning.",
    semantic_revision="straight_running_v2",
)
COMMAND_RANGES = list(RANGES) + [NEW[name]["bounds"] for name in NEW]
COMMAND_RANGES[TASK_NAMES.index("turn")] = TURN_RANGE
HUMAN_HEIGHT = 1.2701193988323212
ROOT_TASKS = [TASK_NAMES.index(name) for name in ("walk", "back_walk", "turn")]


def command_error(predicted, requested, task_indices):
    # Unwrapped turn errors preserve direction and turns greater than pi.
    return predicted - requested


def new_quantity(positions, name):
    """Evaluate staged task quantities, including stationary-running march."""
    root = positions[:, :, 0]
    if name == "squat":
        return root[:, :5, 2].mean(1) - root[:, :, 2].amin(1)
    if name == "bow":
        torso = (positions[:, :, 16] + positions[:, :, 17]) / 2 - root
        return torch.atan2(torso[..., 0], torso[..., 2]).amax(1)
    if name == "clap":
        hands = torch.linalg.vector_norm(
            positions[:, :, 20] - positions[:, :, 21], dim=-1
        )
        return hands.amax(1) - hands.amin(1)
    if name == "point":
        return (positions[:, :, 21, 0] - root[:, :, 0]).amax(1)
    if name == "stretch":
        return (
            (positions[:, :, 20, 2] + positions[:, :, 21, 2]) / 2 - root[:, :, 2]
        ).amax(1)
    if name == "twist":
        lateral = positions[:, :, 16, :2] - positions[:, :, 17, :2]
        pelvis = positions[:, :, 1, :2] - positions[:, :, 2, :2]
        angle = torch.atan2(lateral[..., 1], lateral[..., 0]) - torch.atan2(
            pelvis[..., 1], pelvis[..., 0]
        )
        return torch.atan2(angle.sin(), angle.cos()).abs().amax(1)
    if name == "march":
        feet = positions[:, :, [7, 8], 2]
        # Running crops need not begin in double support. Remove start-phase bias;
        # the legacy release and historical frozen training code remain unchanged.
        baseline = torch.quantile(feet, 0.05, dim=1, keepdim=True)
        return (feet - baseline).amax(1).mean(1)
    if name == "jog":
        return torch.linalg.vector_norm(root[:, 1:, :2] - root[:, :-1, :2], dim=-1).sum(
            1
        ) / ((positions.shape[1] - 1) / 20)
    if name == "arm_circle":
        return (
            positions[:, :, [20, 21], 2].amax(1) - positions[:, :, [20, 21], 2].amin(1)
        ).mean(1)
    raise ValueError(name)


def measure(skeleton, motion, task_indices, lengths):
    # FK is causal in time. Replace invalid padding before the single batched FK:
    # zero rotation padding would otherwise create undefined atan2 gradients.
    frame_indices = torch.arange(motion.shape[1], device=motion.device)[None]
    frame_indices = torch.minimum(frame_indices, lengths[:, None] - 1)
    safe_motion = motion.gather(
        1, frame_indices[..., None].expand(-1, -1, motion.shape[-1])
    )
    positions = skeleton(safe_motion)
    groups = {}
    metadata = torch.stack([task_indices, lengths], dim=1).tolist()
    for sample_index, (task_index, frames) in enumerate(metadata):
        groups.setdefault((task_index, frames), []).append(sample_index)
    measured = []
    order = []
    for (task_index, frames), indices in groups.items():
        name = TASK_NAMES[task_index]
        selected = positions[indices, :frames]
        if name == "turn":
            result = signed_turn(selected)
        elif name in NEW:
            result = new_quantity(selected, name)
        else:
            result = quantity(selected, name, scale=HUMAN_HEIGHT / skeleton.height)
        measured.append(result)
        order.extend(indices)
    inverse_order = [0] * len(order)
    for grouped_index, sample_index in enumerate(order):
        inverse_order[sample_index] = grouped_index
    return torch.cat(measured)[inverse_order]


def measure_commands(skeleton, motion, task_indices, lengths):
    """Return XYZ for reach and [scalar,0,0] for every other active task."""
    from .reach import REACH_INDEX, RETIRED_POINT_INDEX, target_xyz

    if (task_indices == RETIRED_POINT_INDEX).any():
        raise ValueError("point is merged into reach")
    output = motion.new_zeros(len(motion), 3)
    scalar_indices = (task_indices != REACH_INDEX).nonzero(as_tuple=True)[0]
    if len(scalar_indices):
        output[scalar_indices, 0] = measure(
            skeleton,
            motion[scalar_indices],
            task_indices[scalar_indices],
            lengths[scalar_indices],
        )
    for index in (task_indices == REACH_INDEX).nonzero(as_tuple=True)[0].tolist():
        positions = skeleton(motion[index : index + 1, : int(lengths[index])])
        output[index] = target_xyz(positions)[0]
    return output
