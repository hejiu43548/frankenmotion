"""The release's twenty tasks; single-task turn/spin experiments are excluded."""

import torch

from shared_motion.adapter.catalog import NEW, quantities
from shared_motion.adapter.schema import RANGES, TASKS
from shared_motion.adapter.kinematics import quantity
from .turn import TURN_RANGE, signed_turn

TASK_NAMES = TASKS + list(NEW)
COMMAND_RANGES = list(RANGES) + [NEW[name]["bounds"] for name in NEW]
COMMAND_RANGES[TASK_NAMES.index("turn")] = TURN_RANGE
HUMAN_HEIGHT = 1.2701193988323212
ROOT_TASKS = [TASK_NAMES.index(name) for name in ("walk", "back_walk", "turn")]


def command_error(predicted, requested, task_indices):
    # Unwrapped turn errors preserve direction and turns greater than pi.
    return predicted - requested


def measure(skeleton, motion, task_indices, lengths):
    measured = []
    for sample_index, task_index in enumerate(task_indices.tolist()):
        name = TASK_NAMES[task_index]
        positions = skeleton(
            motion[sample_index : sample_index + 1, : int(lengths[sample_index])]
        )
        if name == "turn":
            result = signed_turn(positions)
        elif name in NEW:
            result = quantities(positions)[name]
        else:
            result = quantity(positions, name, scale=HUMAN_HEIGHT / skeleton.height)
        measured.append(result[0])
    return torch.stack(measured)
