"""Reference-only gait-phase selection and velocity-preserving supported bridges."""

from pathlib import Path

import mujoco
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp


def phase_features(model, states):
    data = mujoco.MjData(model)
    foot_ids = [model.body(side + "_ankle_roll_link").id for side in ["left", "right"]]
    positions = []
    for state in states:
        data.qpos[:] = state
        mujoco.mj_forward(model, data)
        positions.append(data.xpos[foot_ids].copy())
    positions = np.array(positions)
    velocities = np.gradient(positions, 0.05, axis=0)
    rotations = Rotation.from_quat(states[:, [4, 5, 6, 3]])
    local_positions = np.stack(
        [
            rotations.inv().apply(positions[:, index] - states[:, :3])
            for index in range(2)
        ],
        axis=1,
    )
    local_velocities = np.stack(
        [rotations.inv().apply(velocities[:, index]) for index in range(2)], axis=1
    )
    support_cost = np.linalg.norm(velocities[:, :, :2], axis=2) + 5 * (
        positions[:, :, 2] - positions[:, :, 2].min(axis=1, keepdims=True)
    )
    support = np.argmin(support_cost, axis=1)
    return dict(local=local_positions, velocity=local_velocities, support=support)


def select_plan(config, model):
    sources = {segment.source for segment in config.plan}
    states = {
        source: np.load(Path(config.output) / "retarget" / (source + ".npz"))["qpos"]
        for source in sources
    }
    features = {
        source: phase_features(model, values) for source, values in states.items()
    }
    pairs = [
        (0, 1, config.gait_phase.walk_end_window, config.gait_phase.turn_start_window),
        (1, 2, config.gait_phase.turn_end_window, config.gait_phase.walk_start_window),
    ]
    selections = []
    for outgoing_index, incoming_index, outgoing_window, incoming_window in pairs:
        outgoing = config.plan[outgoing_index]
        incoming = config.plan[incoming_index]
        outgoing_features = features[outgoing.source]
        incoming_features = features[incoming.source]
        candidates = []
        for outgoing_frame in range(
            round(outgoing_window[0] * 20), round(outgoing_window[1] * 20) + 1
        ):
            for incoming_frame in range(
                round(incoming_window[0] * 20), round(incoming_window[1] * 20) + 1
            ):
                support = int(outgoing_features["support"][outgoing_frame])
                if support != incoming_features["support"][incoming_frame]:
                    continue
                swing = 1 - support
                outgoing_pose = outgoing_features["local"][outgoing_frame]
                incoming_pose = incoming_features["local"][incoming_frame]
                outgoing_velocity = (
                    outgoing_features["velocity"][outgoing_frame] * outgoing.speed
                )
                incoming_velocity = (
                    incoming_features["velocity"][incoming_frame] * incoming.speed
                )
                spreads = [
                    abs(pose[0, 0] - pose[1, 0])
                    for pose in [outgoing_pose, incoming_pose]
                ]
                if max(spreads) > config.gait_phase.max_fore_aft_gap_m:
                    continue
                if min(outgoing_velocity[swing, 0], incoming_velocity[swing, 0]) < 0.15:
                    continue
                score = float(
                    20 * np.sum((outgoing_pose - incoming_pose) ** 2)
                    + 0.15 * np.sum((outgoing_velocity - incoming_velocity) ** 2)
                    + 3 * sum(value**2 for value in spreads)
                )
                candidates.append(
                    dict(
                        score=score,
                        outgoing_frame=outgoing_frame,
                        incoming_frame=incoming_frame,
                        support_index=support,
                        fore_aft_gaps_m=spreads,
                        swing_forward_speeds_m_s=[
                            float(outgoing_velocity[swing, 0]),
                            float(incoming_velocity[swing, 0]),
                        ],
                    )
                )
        if not candidates:
            raise ValueError(
                "No same-support, forward-swing phase pair in configured windows"
            )
        selected = min(candidates, key=lambda candidate: candidate["score"])
        outgoing.crop[1] = selected["outgoing_frame"] / 20
        incoming.crop[0] = selected["incoming_frame"] / 20
        selections.append(
            dict(
                from_name=outgoing.name,
                to_name=incoming.name,
                candidate_count=len(candidates),
                **selected
            )
        )
    # Repeat the chosen walk end / turn crop under the opposite world heading.
    config.plan[2].crop[1] = config.plan[0].crop[1]
    config.plan[3].crop = list(config.plan[1].crop)
    selections.append(
        dict(selections[0], from_name=config.plan[2].name, to_name=config.plan[3].name)
    )
    return selections


def hermite(first, last, first_velocity, last_velocity, fractions, duration):
    weights = fractions[:, None]
    return (
        (2 * weights**3 - 3 * weights**2 + 1) * first
        + (weights**3 - 2 * weights**2 + weights) * duration * first_velocity
        + (-2 * weights**3 + 3 * weights**2) * last
        + (weights**3 - weights**2) * duration * last_velocity
    )


def bridge_states(
    model, outgoing, incoming, seconds, support_index, advance_incoming=False
):
    data = mujoco.MjData(model)
    foot_ids = [model.body(side + "_ankle_roll_link").id for side in ["left", "right"]]

    def feet(state):
        data.qpos[:] = state
        mujoco.mj_forward(model, data)
        return data.xpos[foot_ids].copy()

    first = outgoing[-1]
    incoming_index = round(seconds * 50) if advance_incoming else 0
    last = incoming[incoming_index]
    fractions = np.linspace(0, 1, round(seconds * 50) + 1)
    outgoing_velocity = (outgoing[-1] - outgoing[-2]) / 0.02
    incoming_velocity = (incoming[incoming_index + 1] - incoming[incoming_index]) / 0.02
    bridge = hermite(
        first, last, outgoing_velocity, incoming_velocity, fractions, seconds
    )
    bridge[:, 3:7] = Slerp(
        [0, 1], Rotation.from_quat(np.stack([first, last])[:, [4, 5, 6, 3]])
    )(fractions).as_quat()[:, [3, 0, 1, 2]]
    outgoing_feet = feet(first)
    incoming_feet = feet(last)
    outgoing_foot_velocity = (outgoing_feet - feet(outgoing[-2])) / 0.02
    incoming_foot_velocity = (feet(incoming[incoming_index + 1]) - incoming_feet) / 0.02
    targets = hermite(
        outgoing_feet.ravel(),
        incoming_feet.ravel(),
        outgoing_foot_velocity.ravel(),
        incoming_foot_velocity.ravel(),
        fractions,
        seconds,
    ).reshape(-1, 2, 3)
    targets[:, support_index] = outgoing_feet[support_index]
    indices = np.r_[np.arange(3), np.arange(7, 19)]
    lower = np.r_[[-np.inf] * 3, model.jnt_range[1:13, 0]]
    upper = np.r_[[np.inf] * 3, model.jnt_range[1:13, 1]]
    errors = []
    for frame_index in range(1, len(bridge) - 1):
        desired = bridge[frame_index].copy()

        def residual(values):
            candidate = desired.copy()
            candidate[indices] = values
            return np.r_[
                (feet(candidate) - targets[frame_index]).ravel() * 40,
                (values - desired[indices]) * 0.15,
            ]

        solution = least_squares(
            residual,
            np.clip(desired[indices], lower + 1e-7, upper - 1e-7),
            bounds=(lower, upper),
            max_nfev=40,
        )
        bridge[frame_index, indices] = solution.x
        errors.append(
            float(
                np.linalg.norm(
                    feet(bridge[frame_index])[support_index]
                    - outgoing_feet[support_index]
                )
            )
        )
    return bridge, max(errors)


def phase_resample(states, speed):
    """Clamp floating-point grid endpoints before quaternion interpolation."""
    times = np.arange(len(states)) / 20 / speed
    samples = np.minimum(np.arange(int(times[-1] / 0.02) + 1) * 0.02, times[-1])
    result = np.stack(
        [np.interp(samples, times, column) for column in states.T], axis=1
    )
    result[:, 3:7] = Slerp(times, Rotation.from_quat(states[:, [4, 5, 6, 3]]))(
        samples
    ).as_quat()[:, [3, 0, 1, 2]]
    return result
