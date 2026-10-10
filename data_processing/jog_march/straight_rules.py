"""Straight-path admission independent of the actor's changing body heading."""

import numpy as np
from scipy.ndimage import uniform_filter1d


def straight_metrics(joints, rules):
    root = joints[:, 0, :2].astype(np.float64)
    displacement = root[-1] - root[0]
    distance = np.linalg.norm(displacement)
    direction = displacement / max(distance, 1e-8)
    offset = root - root[0]
    progress = offset @ direction
    perpendicular = offset - progress[:, None] * direction
    path = np.linalg.norm(np.diff(root, axis=0), axis=1).sum()
    lateral = joints[:, 1, :2] - joints[:, 2, :2]
    heading = np.unwrap(np.arctan2(-lateral[:, 0], lateral[:, 1]))
    heading = uniform_filter1d(heading, rules.smooth_frames, mode="nearest")
    velocity = uniform_filter1d(
        np.gradient(root, axis=0), rules.smooth_frames, axis=0, mode="nearest"
    )
    direction_angles = np.unwrap(np.arctan2(velocity[:, 1], velocity[:, 0]))
    metrics = dict(
        net_to_path_ratio=float(distance / max(path, 1e-8)),
        maximum_line_deviation_m=float(np.linalg.norm(perpendicular, axis=1).max()),
        body_heading_span_deg=float(np.rad2deg(np.ptp(heading))),
        body_heading_net_deg=float(np.rad2deg(abs(heading[-1] - heading[0]))),
        travel_direction_span_deg=float(np.rad2deg(np.ptp(direction_angles))),
    )
    failures = []
    if metrics["net_to_path_ratio"] < rules.minimum_net_to_path_ratio:
        failures.append("curved_or_backtracking_path")
    if metrics["maximum_line_deviation_m"] > rules.maximum_line_deviation_m:
        failures.append("deviates_from_start_end_line")
    if metrics["body_heading_span_deg"] > rules.maximum_body_heading_span_deg:
        failures.append("body_heading_turn_inside_clip")
    if metrics["body_heading_net_deg"] > rules.maximum_body_heading_net_deg:
        failures.append("body_heading_changed_at_end")
    if metrics["travel_direction_span_deg"] > rules.maximum_travel_direction_span_deg:
        failures.append("travel_direction_turn_inside_clip")
    return metrics, failures
