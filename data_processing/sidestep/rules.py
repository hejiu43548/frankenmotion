"""Physical sidestep admission and a reversible SMPL body mirror."""

import numpy as np

from data_processing.sidestep.inspect_sources import movement_signals

BODY_SWAP = np.array(
    [0, 2, 1, 3, 5, 4, 6, 8, 7, 9, 11, 10, 12, 14, 13, 15, 17, 16, 19, 18, 21, 20]
)


def mirror_body(poses, translation):
    """Reflect world X and SMPL rest X; swap anatomical L/R body joints.

    Axial vectors transform as det(reflection)*reflection. Only the first22
    body joints are used by the205-feature pipeline; finger poses are excluded.
    The fixed asymmetric skeleton is re-FK'd by the standard converter.
    """
    mirrored_poses = np.asarray(poses[:, :66]).reshape(-1, 22, 3)[:, BODY_SWAP].copy()
    mirrored_poses[:, :, 1:] *= -1
    mirrored_translation = np.asarray(translation).copy()
    mirrored_translation[:, 0] *= -1
    return mirrored_poses.reshape(-1, 66), mirrored_translation


def check_clip(joints, expected_direction, policy):
    signals = movement_signals(joints)
    direction = 1 if expected_direction == "left" else -1
    root = joints[:, 0]
    lateral = root[:, 1] - root[0, 1]
    forward = root[:, 0] - root[0, 0]
    directed = direction * lateral
    increments = np.diff(directed)
    total_path = np.abs(increments).sum()
    separation = signals["ankle_separation"]
    metrics = dict(
        direction=expected_direction,
        signed_endpoint_lateral_m=float(lateral[-1]),
        directed_distance_m=float(directed[-1]),
        forward_excursion_m=float(np.abs(forward).max()),
        heading_change_rad=float(np.ptp(signals["heading"])),
        reverse_fraction=float(
            -np.minimum(increments, 0).sum() / max(total_path, 1e-8)
        ),
        minimum_ankle_separation_m=float(separation.min()),
        opening_excursion_m=float(
            separation.max() - max(separation[0], separation[-1])
        ),
        initial_ankle_separation_m=float(separation[0]),
        final_ankle_separation_m=float(separation[-1]),
        left_ankle_lift_m=float(np.ptp(joints[:, 7, 2])),
        right_ankle_lift_m=float(np.ptp(joints[:, 8, 2])),
        forward_to_lateral_ratio=float(np.abs(forward).max() / max(directed[-1], 1e-8)),
    )
    failures = []
    if metrics["directed_distance_m"] < policy.minimum_clip_travel_m:
        failures.append("wrong_direction_or_too_little_lateral_travel")
    if metrics["forward_to_lateral_ratio"] > policy.maximum_forward_excursion_ratio:
        failures.append("excess_forward_backward_travel")
    if metrics["heading_change_rad"] > policy.maximum_heading_change_rad:
        failures.append("turning_instead_of_sidestep")
    if metrics["reverse_fraction"] > policy.maximum_reverse_fraction:
        failures.append("mixed_lateral_directions")
    if metrics["minimum_ankle_separation_m"] < policy.minimum_ankle_separation_m:
        failures.append("crossing_feet")
    if metrics["opening_excursion_m"] < policy.minimum_opening_excursion_m:
        failures.append("no_open_close_lateral_step")
    if max(separation[0], separation[-1]) > policy.maximum_endpoint_ankle_separation_m:
        failures.append("incomplete_open_close_cycle")
    if (
        max(metrics["left_ankle_lift_m"], metrics["right_ankle_lift_m"])
        > policy.maximum_ankle_lift_m
    ):
        failures.append("high_leg_or_hopping_motion")
    return metrics, failures
