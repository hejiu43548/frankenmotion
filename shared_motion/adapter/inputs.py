import torch
import math
from .schema import TASKS, RANGES, features

REFERENCE_HUMAN_HEIGHT = 1.2701193988323212


def encode_control(values, valid):
    if valid.ndim == 2:
        valid = valid[..., None].expand_as(values)
    return torch.cat(
        (values / values.new_tensor([3.0, math.pi]) * valid, valid.to(values)), -1
    )


KINDS = TASKS + [
    "walk_endpoint",
    "place_hold_retract",
    "back_departure",
    "side_departure",
    "root_profile",
    "turn_endpoint",
]


def make(
    kind, command_value, phase, extra=None, frames=None, skeleton_height=1.372592926
):
    batch_size, num_frames = phase.shape
    frames = frames or (
        60 if kind in ["raise_hand", "reach", "strike", "kick", "jump", "lean"] else 120
    )
    device = phase.device
    root = None
    goal = None
    height = None
    intent = 0
    task_index = TASKS.index(kind) if kind in TASKS else 11
    if kind in ["walk", "back_walk", "turn", "turn_endpoint"]:
        task_index = 4 if kind == "turn_endpoint" else task_index
        intent = 1 if kind == "turn_endpoint" else 0
        root_velocity = torch.zeros(batch_size, num_frames, 2, device=device)
        if task_index == 4:
            root_velocity[:, :, 1] = -command_value[:, None] / ((frames - 1) / 20)
        else:
            root_velocity[:, :, 0] = (
                command_value[:, None] * skeleton_height / REFERENCE_HUMAN_HEIGHT
            )
        root = encode_control(
            root_velocity,
            torch.ones(batch_size, num_frames, device=device, dtype=torch.bool),
        )
    if kind == "walk_endpoint":
        task_index = 10
        intent = 1
        goal = extra
        goal_distance, angle = goal.unbind(-1)
        command_value = goal_distance / 4.3 * REFERENCE_HUMAN_HEIGHT / skeleton_height
        frame_times = phase * 5.95
        grid = torch.arange(120, device=device) * 0.05
        profile = torch.clamp((grid - 0.6) / 0.6, 0, 1) * torch.clamp(
            (5.3 - grid) / 0.8, 0, 1
        )
        profile_integral = profile[:-1].sum() * 0.05
        root_velocity = torch.zeros(batch_size, num_frames, 2, device=device)
        root_velocity[:, :, 0] = (
            goal_distance[:, None]
            * torch.clamp((frame_times - 0.6) / 0.6, 0, 1)
            * torch.clamp((5.3 - frame_times) / 0.8, 0, 1)
            / profile_integral
        )
        root_velocity[:, :, 1] = (
            angle[:, None] * (frame_times < 1.2).to(root_velocity) / 1.2
        )
        root = encode_control(
            root_velocity,
            torch.ones(batch_size, num_frames, device=device, dtype=torch.bool),
        )
    if kind == "place_hold_retract":
        task_index = 1
        intent = 2
        height = extra
    if kind in ["back_departure", "side_departure"]:
        task_index = 6 if kind == "back_departure" else 5
        intent = 3
    if kind == "root_profile":
        root = extra
    return features(
        task_index, command_value, phase, intent, root, goal, height, frames
    )
