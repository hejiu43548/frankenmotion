"""Five generated segments, explicit reference processing, continuous torque rollout."""

import json
import os
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
from omegaconf import OmegaConf
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation, Slerp

from g1_demo_physics import endpoint_ease, place, scene, simulate
from g1_demo2_pipeline import audit, render
from g1_demo2_gait import bridge_states, phase_resample, select_plan


def heading(states):
    return np.unwrap(Rotation.from_quat(states[:, [4, 5, 6, 3]]).as_euler("xyz")[:, 2])


def compose(config):
    model, _ = scene(config)
    data = mujoco.MjData(model)
    foot_ids = [model.body(f"{side}_ankle_roll_link").id for side in ["left", "right"]]

    def feet(state):
        data.qpos[:] = state
        mujoco.mj_forward(model, data)
        return data.xpos[foot_ids].copy()

    phase_selection = select_plan(config, model)
    result = None
    segments = []
    transitions = []
    transformations = []
    for segment_index, segment in enumerate(config.plan):
        original = np.load(Path(config.output) / "retarget" / f"{segment.source}.npz")[
            "qpos"
        ]
        source_start, source_end = [round(value * 20) for value in segment.crop]
        states = phase_resample(original[source_start : source_end + 1], segment.speed)
        eased = endpoint_ease(states)
        if segment_index != 4:
            states[:15] = eased[:15]
        if segment_index != 3:
            states[-15:] = eased[-15:]
        initial_yaw = heading(states[:1])[0]
        states = place(states, -initial_yaw, [0, 0, 0])
        initial_xy = states[0, :2].copy()
        states[:, :2] -= initial_xy
        # Optional gradual heading correction rotates each complete pose about its
        # pelvis. It modifies reference only and is explicitly recorded.
        correction = float(segment.get("heading_correction_deg", 0))
        target_yaw = segment.get("target_yaw_deg")
        if target_yaw is not None:
            correction = float(target_yaw) - float(
                np.rad2deg(heading(states)[-1] - heading(states)[0])
            )
        profile = segment.get("heading_profile_deg")
        if correction or profile is not None:
            fractions = np.linspace(0, 1, len(states))
            if profile is not None:
                knots = np.linspace(0, 1, len(profile))
                offsets = np.interp(fractions, knots, np.deg2rad(list(profile)))
                for knot_index in range(len(knots) - 1):
                    selected = (fractions >= knots[knot_index]) & (
                        fractions <= knots[knot_index + 1]
                    )
                    local_fraction = (fractions[selected] - knots[knot_index]) / (
                        knots[knot_index + 1] - knots[knot_index]
                    )
                    blend = 3 * local_fraction**2 - 2 * local_fraction**3
                    offsets[selected] = np.deg2rad(
                        profile[knot_index] * (1 - blend)
                        + profile[knot_index + 1] * blend
                    )
            else:
                offsets = np.deg2rad(correction) * (3 * fractions**2 - 2 * fractions**3)
            displacements = np.diff(states[:, :3], axis=0)
            displacements = Rotation.from_euler("z", offsets[:-1, None]).apply(
                displacements
            )
            states[1:, :3] = states[0, :3] + np.cumsum(displacements, axis=0)
            for frame_index, offset in enumerate(offsets):
                states[frame_index, 3:7] = (
                    Rotation.from_euler("z", offset)
                    * Rotation.from_quat(states[frame_index, [4, 5, 6, 3]])
                ).as_quat()[[3, 0, 1, 2]]
        stationary_error = None
        if segment.get("stationary_feet", False):
            source_yaw = heading(states)
            target_feet = feet(states[0])
            indices = np.r_[np.arange(3), np.arange(7, 19)]
            lower = np.r_[[-np.inf] * 3, model.jnt_range[1:13, 0]]
            upper = np.r_[[np.inf] * 3, model.jnt_range[1:13, 1]]
            errors = []
            for frame_index in range(len(states)):
                desired = states[frame_index].copy()
                desired[:2] = states[0, :2]
                desired[3:7] = (
                    Rotation.from_euler("z", -source_yaw[frame_index])
                    * Rotation.from_quat(desired[[4, 5, 6, 3]])
                ).as_quat()[[3, 0, 1, 2]]

                def stationary_residual(values):
                    candidate = desired.copy()
                    candidate[indices] = values
                    return np.r_[
                        (feet(candidate) - target_feet).ravel() * 40,
                        (values - desired[indices]) * 0.15,
                    ]

                solution = least_squares(
                    stationary_residual,
                    np.clip(desired[indices], lower + 1e-7, upper - 1e-7),
                    bounds=(lower, upper),
                    max_nfev=40,
                )
                desired[indices] = solution.x
                states[frame_index] = desired
                errors.append(float(np.abs(feet(desired) - target_feet).max()))
            stationary_error = max(errors)
        if segment.get("reverse_prefix", False):
            states = np.concatenate([states[::-1], states[1:]])
        if segment.get("reverse_recovery", False):
            states = np.concatenate([states, states[-2::-1]])
        transform = dict(
            name=segment.name,
            source=segment.source,
            source_crop_frames=[source_start, source_end],
            source_fps=20,
            output_fps=50,
            speed=float(segment.speed),
            normalized_yaw_rad=float(initial_yaw),
            normalized_translation_xy=initial_xy.tolist(),
            heading_correction_deg=correction,
            target_yaw_deg=target_yaw,
            reverse_recovery=bool(segment.get("reverse_recovery", False)),
        )
        transform.update(
            heading_profile_deg=None if profile is None else list(profile),
            reverse_prefix=bool(segment.get("reverse_prefix", False)),
            stationary_feet=bool(segment.get("stationary_feet", False)),
            stationary_max_position_error_m=stationary_error,
        )
        if result is None:
            result = states
            start = 0.0
        else:
            previous = result[-1]
            previous_yaw = heading(result[-1:])[0]
            states = place(states, previous_yaw, [0, 0, 0])
            # Choose the lower outgoing foot, rather than a demo-specific fixed side.
            outgoing_feet = feet(previous)
            incoming_feet = feet(states[0])
            moving_bridge = segment_index == 4
            transition_seconds = float(
                config.gait_phase.transition_seconds
                if moving_bridge
                else config.transition_seconds
            )
            support_index = (
                phase_selection["support_index"]
                if moving_bridge
                else int(np.argmin(outgoing_feet[:, 2]))
            )
            shift = outgoing_feet[support_index] - incoming_feet[support_index]
            states[:, :3] += shift
            incoming_feet = feet(states[0])
            if moving_bridge:
                bridge, plant_error = bridge_states(
                    model, result, states, transition_seconds, support_index
                )
                errors = [plant_error]
            else:
                frame_count = round(transition_seconds * 50)
                fractions = np.linspace(0, 1, frame_count + 1)
                smooth = fractions**3 * (10 - 15 * fractions + 6 * fractions**2)
                bridge = (
                    previous[None] * (1 - smooth[:, None])
                    + states[:1] * smooth[:, None]
                )
                bridge[:, 3:7] = Slerp(
                    [0, 1],
                    Rotation.from_quat(
                        np.stack([previous, states[0]])[:, [4, 5, 6, 3]]
                    ),
                )(smooth).as_quat()[:, [3, 0, 1, 2]]
                indices = np.r_[np.arange(3), np.arange(7, 19)]
                lower = np.r_[[-np.inf] * 3, model.jnt_range[1:13, 0]]
                upper = np.r_[[np.inf] * 3, model.jnt_range[1:13, 1]]
                errors = []
                for frame_index in range(1, frame_count):
                    desired = bridge[frame_index].copy()
                    targets = (
                        outgoing_feet * (1 - smooth[frame_index])
                        + incoming_feet * smooth[frame_index]
                    )
                    targets[support_index] = outgoing_feet[support_index]
                    targets[1 - support_index, 2] += (
                        0.025 * np.sin(np.pi * fractions[frame_index]) ** 2
                    )

                    def residual(values):
                        candidate = desired.copy()
                        candidate[indices] = values
                        return np.r_[
                            (feet(candidate) - targets).ravel() * 40,
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
            transition_start = (len(result) - 1) * 0.02
            result = np.concatenate([result, bridge[1:], states[1:]])
            start = transition_start + transition_seconds
            transitions.append(
                dict(
                    start=transition_start,
                    end=start,
                    from_name=segments[-1]["name"],
                    to_name=segment.name,
                    bridge_method=(
                        "phase_matched_hermite"
                        if moving_bridge
                        else "stationary_quintic"
                    ),
                    planted_foot=["left", "right"][support_index],
                    maximum_reference_plant_error_m=max(errors),
                )
            )
            transform.update(
                world_yaw_rad=float(previous_yaw), world_translation=shift.tolist()
            )
        segments.append(
            dict(name=segment.name, start=start, end=(len(result) - 1) * 0.02)
        )
        transformations.append(transform)
    result = np.concatenate(
        [result, np.repeat(result[-1:], round(config.final_hold_seconds * 50), axis=0)]
    )
    destination = Path(config.output) / "attempts" / config.attempt
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "actual.npz").exists():
        raise FileExistsError(destination)
    np.savez_compressed(destination / "reference.npz", qpos=result, fps=50.0)
    (destination / "composition.json").write_text(
        json.dumps(
            dict(
                segments=segments,
                transitions=transitions,
                transformations=transformations,
                phase_selection=phase_selection,
                endpoint_ease_seconds=0.3,
                moving_boundary_eased=False,
                final_hold_seconds=float(config.final_hold_seconds),
            ),
            indent=2,
        )
    )
    OmegaConf.save(config, destination / "config.yaml")
    simulate(config, result, destination)
    audit(config)


def run_phase(config):
    if config.phase == "compose":
        compose(config)
    elif config.phase == "audit":
        audit(config)
    elif config.phase == "render":
        render(config)
    elif config.phase == "verify":
        from g1_demo2_verify import verify

        verify(config)
        from g1_demo2_phase_audit import compare

        compare(
            Path(config.output) / "attempts" / config.attempt,
            config.gait_phase.baseline,
        )
    else:
        raise ValueError(config.phase)
