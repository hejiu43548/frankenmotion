"""World-fixed wrist commands shared by reach and strike.

A caller converts an initial-body-relative command exactly once using an external
initial pose. The resulting world point is never rebuilt from a generated pose.
World axes follow the dataset's initial hip-yaw canonicalization, meters, Z up.
"""

import torch
from torch import nn
from .strike_residual_policy import apply_residual


def body_basis(joints):
    left = joints[..., 1, :] - joints[..., 2, :]
    left = torch.cat([left[..., :2], torch.zeros_like(left[..., 2:])], -1)
    width = left.norm(dim=-1, keepdim=True)
    if torch.any(width < 1e-6):
        raise ValueError("Degenerate hip axis")
    left = left / width
    up = torch.zeros_like(left)
    up[..., 2] = 1
    return torch.stack([torch.linalg.cross(left, up, dim=-1), left, up], -1)


def freeze_initial_target(relative_xyz, initial_joints):
    """Convert once, before generation, from an externally supplied initial pose."""
    if (
        not torch.isfinite(relative_xyz).all()
        or not torch.isfinite(initial_joints).all()
    ):
        raise ValueError("Initial target and pose must be finite")
    if relative_xyz.shape != initial_joints.shape[:-2] + (3,):
        raise ValueError("Target and initial pose batch shapes must match")
    return initial_joints[..., 0, :] + (
        body_basis(initial_joints) @ relative_xyz[..., None]
    ).squeeze(-1)


def selected_wrists(joints, hands):
    if hands.shape != (len(joints),) or torch.any((hands < 0) | (hands > 1)):
        raise ValueError("hands must be [batch], 0=left or 1=right")
    batch_indices = torch.arange(len(joints), device=joints.device)
    return joints[
        batch_indices[:, None],
        torch.arange(joints.shape[1], device=joints.device)[None],
        (20 + hands)[:, None],
    ]


def fixed_observations(motion, world_target, event_frames, hands, skeleton):
    joints = skeleton(motion)
    indices = torch.arange(len(motion), device=motion.device)
    event_joints = joints[indices, event_frames]
    basis = body_basis(event_joints)
    wrist = event_joints[indices, 20 + hands] - event_joints[:, 0]
    target = world_target - event_joints[:, 0]
    wrist_local = (basis.transpose(-1, -2) @ wrist[..., None]).squeeze(-1)
    target_local = (basis.transpose(-1, -2) @ target[..., None]).squeeze(-1)
    rotations = motion[..., 4:136].reshape(*motion.shape[:2], 22, 6)
    control = torch.tensor([[13, 16, 18], [14, 17, 19]], device=motion.device)[hands]
    selected = rotations[indices, event_frames][indices[:, None], control].flatten(1)
    return torch.cat(
        [
            selected,
            wrist_local,
            target_local,
            (target_local - wrist_local) / 0.2,
            torch.nn.functional.one_hot(hands, 2).to(motion.dtype),
        ],
        -1,
    )


class FixedTargetPolicy(nn.Module):
    def __init__(self, width=128):
        super().__init__()
        self.mean = nn.Sequential(
            nn.Linear(29, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 9),
        )
        nn.init.zeros_(self.mean[-1].weight)
        nn.init.zeros_(self.mean[-1].bias)
        self.log_std = nn.Parameter(torch.full((9,), -1.5))

    def distribution(self, observation):
        return torch.distributions.Normal(
            self.mean(observation), self.log_std.clamp(-3, -0.3).exp()
        )


def apply_fixed_residual(
    motion, action, event_frames, hands, skeleton, max_angle=0.35, radius_frames=12
):
    return apply_residual(
        motion, action, event_frames, skeleton, max_angle, radius_frames, hands=hands
    )


def fixed_metrics(joints, world_target, hands, event_frames, task, fps=20):
    """Geometric diagnostics at the scheduled event and across its local window.

    No target follows the pelvis; no best-frame selection replaces event error.
    Strike speed is measured toward the fixed target before the scheduled event.
    Reach dwell uses five consecutive frames within 10 cm in the event window.
    These are kinematic checks, not physical-contact certification.
    """
    if task not in ("reach", "strike"):
        raise ValueError("Expected reach or strike")
    wrists = selected_wrists(joints, hands)
    distances = (wrists - world_target[:, None]).norm(dim=-1)
    indices = torch.arange(len(joints), device=joints.device)
    frame_ids = torch.arange(joints.shape[1], device=joints.device)[None]
    window = (frame_ids - event_frames[:, None]).abs() <= 8
    errors = distances[indices, event_frames]
    if task == "reach":
        errors = (wrists[:, -5:].mean(1) - world_target).norm(dim=-1)
    min_error = distances.masked_fill(~window, float("inf")).min(-1).values
    velocity = (wrists[:, 1:] - wrists[:, :-1]) * fps
    direction = torch.nn.functional.normalize(
        world_target[:, None] - wrists[:, :-1], dim=-1
    )
    approach = (velocity * direction).sum(-1)
    pre_event = (frame_ids[:, :-1] >= event_frames[:, None] - 8) & (
        frame_ids[:, :-1] < event_frames[:, None]
    )
    approach_speed = approach.masked_fill(~pre_event, -float("inf")).max(-1).values
    within = (distances < 0.1) & window
    dwell = (
        (distances[:, -5:] < 0.1).all(-1)
        if task == "reach"
        else within.unfold(1, 5, 1).all(-1).any(-1)
    )
    hold_speed = velocity[:, -4:].norm(dim=-1).mean(-1)
    root_excursion = (joints[:, :, 0] - joints[:, :1, 0]).norm(dim=-1).max(-1).values
    peak_speed = velocity.norm(dim=-1).max(-1).values
    # Re-use a predeclared quality gate, rather than modifying rewards after eval.
    quality = (root_excursion <= 0.4) & (peak_speed <= 8)
    strict = (
        (errors < 0.1)
        & quality
        & (
            (approach_speed >= 0.8)
            if task == "strike"
            else dwell & (hold_speed <= 0.35)
        )
    )
    return dict(
        error_m=errors,
        window_min_error_m=min_error,
        approach_speed_m_s=approach_speed,
        dwell_5_frames=dwell,
        final_hold_speed_m_s=hold_speed,
        root_excursion_m=root_excursion,
        peak_wrist_speed_m_s=peak_speed,
        strict_pass=strict,
    )
