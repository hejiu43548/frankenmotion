import torch
import math

TASKS = [
    "raise_hand",
    "reach",
    "strike",
    "wave",
    "turn",
    "sidestep",
    "back_walk",
    "kick",
    "jump",
    "lean",
    "walk",
]
RANGES = [
    (0.35, 0.75),
    (0.25, 0.55),
    (1.5, 2.5),
    (0.08, 0.22),
    (0.45, 1.5),
    (0.4, 1.2),
    (0.35, 0.9),
    (0.25, 0.7),
    (0.25, 0.55),
    (0.4, 0.9),
    (0.5, 1.1),
]
# All fields are semantic inputs, never checkpoint IDs or expert selectors.
FIELDS = {
    "task_onehot": (0, 12),
    "intent_onehot": (12, 16),
    "task_value": 16,
    "task_valid": 17,
    "root_speed": 18,
    "root_yaw_rate": 19,
    "root_valid": (20, 22),
    "goal_distance": 22,
    "goal_sin": 23,
    "goal_cos": 24,
    "goal_valid": 25,
    "reach_height": 26,
    "reach_height_valid": 27,
    "time_features": (28, 45),
    "duration": 45,
}
INTENTS = [
    "free_motion",
    "endpoint_motion",
    "place_hold_retract",
    "start_stop_departure",
]
COMMAND_FEATURE_COUNT = 46


def features(
    task_index,
    command_value,
    phase,
    intent=0,
    root=None,
    goal=None,
    height=None,
    frames=120,
    task_valid=True,
):
    device = phase.device
    batch_size, num_frames = phase.shape
    command_features = torch.zeros(
        batch_size, num_frames, COMMAND_FEATURE_COUNT, device=device
    )
    command_features[:, :, task_index] = 1
    command_features[:, :, 12 + intent] = 1
    if task_index < 11 and task_valid:
        lower_bound, upper_bound = RANGES[task_index]
        command_features[:, :, 16] = (
            (command_value - lower_bound) / (upper_bound - lower_bound) * 2 - 1
        ).clamp(-5, 5)[:, None]
        command_features[:, :, 17] = 1
    if root is not None:
        command_features[:, :, 18:22] = root
    if goal is not None:
        command_features[:, :, 22] = goal[:, 0, None] / 3
        command_features[:, :, 23] = goal[:, 1, None].sin()
        command_features[:, :, 24] = goal[:, 1, None].cos()
        command_features[:, :, 25] = 1
    if height is not None:
        command_features[:, :, 26] = (height[:, None] - 0.84) / 0.03
        command_features[:, :, 27] = 1
    frequencies = phase.new_tensor([1, 2, 3, 4, 6, 8, 10, 12])
    phase_angles = phase[..., None] * frequencies * 2 * math.pi
    command_features[:, :, 28:45] = torch.cat(
        [phase[..., None], phase_angles.sin(), phase_angles.cos()], -1
    )
    command_features[:, :, 45] = frames / 120
    return command_features


def availability(command_features):
    return (
        command_features[..., 17:18]
        .maximum(command_features[..., 20:22].amax(-1, keepdim=True))
        .maximum(command_features[..., 25:26])
        .maximum(command_features[..., 27:28])
    )
