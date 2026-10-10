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


def compare(folder, baseline):
    report = dict(baseline=measure(baseline), candidate=measure(folder))
    candidate = report["candidate"]["transitions"]
    report["improvement_checks"] = dict(
        all_transitions_keep_support=all(
            item["unsupported_seconds"] == 0 for item in candidate
        ),
        no_wide_stance_pause=all(
            item["wide_stance_pause_s"] == 0 for item in candidate
        ),
        no_stationary_bridge=all(item["stationary_pause_s"] == 0 for item in candidate),
        compact_feet_during_bridges=all(
            item["maximum_fore_aft_gap_m"] < 0.20 for item in candidate
        ),
        nearby_pause_below_80ms=all(
            item["nearby_longest_stationary_pause_s"] <= 0.08 for item in candidate
        ),
        both_walk_turn_root_speeds_above_20cm_s=all(
            item["root_speed_min_mean_m_s"][0] > 0.20
            for item in candidate
            if item["from_name"].startswith("walk")
        ),
    )
    report["passed"] = all(report["improvement_checks"].values())
    (Path(folder) / "phase_comparison.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    compare(sys.argv[1], sys.argv[2])
