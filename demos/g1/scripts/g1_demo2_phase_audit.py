"""Compare actual gait transitions with the preserved pre-phase baseline."""

import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation


def longest_duration(mask):
    padded = np.r_[False, mask, False].astype(int)
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1)
    return float(max(ends - starts, default=0) * 0.02)


def measure(folder):
    folder = Path(folder)
    archive = np.load(folder / "actual.npz")
    states = archive["qpos"]
    names = archive["body_names"].tolist()
    composition = json.loads((folder / "composition.json").read_text())
    yaw = Rotation.from_quat(states[:, [4, 5, 6, 3]]).as_euler("xyz")[:, 2]
    inverse = Rotation.from_euler("z", -yaw[:, None])
    foot_ids = [names.index(side + "_ankle_roll_link") for side in ["left", "right"]]
    feet = archive["body_pos"][:, foot_ids]
    velocities = np.gradient(feet, 0.02, axis=0)
    gap = np.abs(inverse.apply(feet[:, 0] - feet[:, 1])[:, 0])
    root_speed = np.linalg.norm(archive["qvel"][:, :2], axis=1)
    foot_speed = np.linalg.norm(velocities[:, :, :2], axis=2)
    swing_velocities = np.stack(
        [inverse.apply(velocities[:, index])[:, 0] for index in [0, 1]], axis=1
    )
    records = []
    for transition in composition["transitions"]:
        first = round(transition["start"] * 50)
        last = min(len(states) - 2, round(transition["end"] * 50))
        indices = slice(first, last + 1)
        swing = 0 if transition["planted_foot"] == "right" else 1
        stationary = (root_speed < 0.12) & (foot_speed.max(axis=1) < 0.25)
        wide_pause = stationary & (gap > 0.25)
        nearby = slice(max(0, first - 10), min(len(states), last + 11))
        records.append(
            dict(
                from_name=transition["from_name"],
                to_name=transition["to_name"],
                interval_s=[first / 50, last / 50],
                foot_fore_aft_gap_start_end_m=[float(gap[first]), float(gap[last])],
                maximum_fore_aft_gap_m=float(gap[indices].max()),
                root_speed_min_mean_m_s=[
                    float(root_speed[indices].min()),
                    float(root_speed[indices].mean()),
                ],
                swing_forward_velocity_min_max_m_s=[
                    float(swing_velocities[indices, swing].min()),
                    float(swing_velocities[indices, swing].max()),
                ],
                swing_forward_travel_m=float(
                    np.maximum(swing_velocities[indices, swing], 0).sum() * 0.02
                ),
                swing_backward_travel_m=float(
                    -np.minimum(swing_velocities[indices, swing], 0).sum() * 0.02
                ),
                wide_stance_pause_s=float(wide_pause[indices].sum() * 0.02),
                stationary_pause_s=float(stationary[indices].sum() * 0.02),
                nearby_longest_wide_stance_pause_s=longest_duration(wide_pause[nearby]),
                nearby_longest_stationary_pause_s=longest_duration(stationary[nearby]),
                support_min_N=float(
                    archive["telemetry"][first:last, :2].sum(axis=1).min()
                ),
                unsupported_seconds=float(
                    archive["telemetry"][first:last, 8].sum() * 0.002
                ),
            )
        )
    return dict(
        attempt=folder.name,
        definitions=dict(
            wide_fore_aft_gap_m=0.25,
            stationary_root_speed_below_m_s=0.12,
            stationary_both_foot_speed_below_m_s=0.25,
            nearby_margin_s=0.2,
        ),
        transitions=records,
    )


def return_metrics(folder):
    folder = Path(folder)
    archive = np.load(folder / "actual.npz")
    states = archive["qpos"]
    composition = json.loads((folder / "composition.json").read_text())
    segment = next(
        segment for segment in composition["segments"] if segment["name"] == "walk_back"
    )
    first, last = [round(segment[key] * 50) for key in ["start", "end"]]
    yaw = np.rad2deg(
        np.unwrap(Rotation.from_quat(states[:, [4, 5, 6, 3]]).as_euler("xyz")[:, 2])
    )
    portion = states[first : last + 1]
    return dict(
        interval_s=[first / 50, last / 50],
        yaw_start_end_deg=[float(yaw[first]), float(yaw[last])],
        yaw_min_max_deg=[
            float(yaw[first : last + 1].min()),
            float(yaw[first : last + 1].max()),
        ],
        yaw_range_deg=float(np.ptp(yaw[first : last + 1])),
        final_distance_m=float(np.linalg.norm(states[-1, :2] - states[0, :2])),
        net_walk_distance_m=float(np.linalg.norm(portion[-1, :2] - portion[0, :2])),
        path_m=float(np.linalg.norm(np.diff(portion[:, :2], axis=0), axis=1).sum()),
        final_hold_drift_m=float(np.linalg.norm(states[-1, :2] - states[last, :2])),
    )


def compare(folder, baseline):
    report = dict(
        baseline=measure(baseline),
        candidate=measure(folder),
        return_baseline=return_metrics(baseline),
        return_candidate=return_metrics(folder),
    )
    transition = next(
        item
        for item in report["candidate"]["transitions"]
        if item["to_name"] == "walk_back"
    )
    report["improvement_checks"] = dict(
        return_bridge_supported=transition["unsupported_seconds"] == 0,
        no_return_bridge_pause=transition["stationary_pause_s"] == 0,
        nearby_pause_under_80ms=transition["nearby_longest_stationary_pause_s"] <= 0.08,
        compact_return_bridge=transition["maximum_fore_aft_gap_m"] < 0.2,
        return_heading_range_reduced=report["return_candidate"]["yaw_range_deg"]
        < report["return_baseline"]["yaw_range_deg"] * 0.6,
    )
    report["passed"] = all(report["improvement_checks"].values())
    (Path(folder) / "phase_comparison.json").write_text(json.dumps(report, indent=2))
    print(
        json.dumps(
            dict(
                checks=report["improvement_checks"],
                return_motion=report["return_candidate"],
                transition=transition,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    compare(sys.argv[1], sys.argv[2])
