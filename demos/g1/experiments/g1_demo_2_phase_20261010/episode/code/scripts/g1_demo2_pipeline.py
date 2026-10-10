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

from g1_demo_physics import EFFORT, endpoint_ease, place, resample, scene, simulate


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

    result = None
    segments = []
    transitions = []
    transformations = []
    for segment in config.plan:
        original = np.load(Path(config.output) / "retarget" / f"{segment.source}.npz")[
            "qpos"
        ]
        source_start, source_end = [round(value * 20) for value in segment.crop]
        states = endpoint_ease(
            resample(original[source_start : source_end + 1], segment.speed)
        )
        initial_yaw = heading(states[:1])[0]
        states = place(states, -initial_yaw, [0, 0, 0])
        initial_xy = states[0, :2].copy()
        states[:, :2] -= initial_xy
        # Optional gradual heading correction rotates each complete pose about its
        # pelvis. It modifies reference only and is explicitly recorded.
        correction = float(segment.get("heading_correction_deg", 0))
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
            support_index = int(np.argmin(outgoing_feet[:, 2]))
            shift = outgoing_feet[support_index] - incoming_feet[support_index]
            states[:, :3] += shift
            incoming_feet = feet(states[0])
            frame_count = round(config.transition_seconds * 50)
            fractions = np.linspace(0, 1, frame_count + 1)
            smooth = fractions**3 * (10 - 15 * fractions + 6 * fractions**2)
            bridge = (
                previous[None] * (1 - smooth[:, None]) + states[:1] * smooth[:, None]
            )
            bridge[:, 3:7] = Slerp(
                [0, 1],
                Rotation.from_quat(np.stack([previous, states[0]])[:, [4, 5, 6, 3]]),
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
            start = transition_start + config.transition_seconds
            transitions.append(
                dict(
                    start=transition_start,
                    end=start,
                    from_name=segments[-1]["name"],
                    to_name=segment.name,
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
                endpoint_ease_seconds=0.3,
                final_hold_seconds=float(config.final_hold_seconds),
            ),
            indent=2,
        )
    )
    OmegaConf.save(config, destination / "config.yaml")
    simulate(config, result, destination)
    audit(config)


def audit(config):
    destination = Path(config.output) / "attempts" / config.attempt
    archive = np.load(destination / "actual.npz")
    states = archive["qpos"]
    reference = archive["reference"]
    telemetry = archive["telemetry"]
    composition = json.loads((destination / "composition.json").read_text())
    result = json.loads((destination / "result.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(destination / "scene.xml"))
    names = archive["body_names"].tolist()
    yaw = heading(states)
    return_distance = float(np.linalg.norm(states[-1, :2] - states[0, :2]))
    heading_error = float(
        abs(np.rad2deg(np.angle(np.exp(1j * (yaw[-1] - yaw[0] - np.pi)))))
    )
    segment_metrics = []
    for segment in composition["segments"]:
        first = round(segment["start"] * 50)
        last = min(len(states) - 1, round(segment["end"] * 50))
        if first >= last:
            continue
        portion = states[first : last + 1]
        rotations = Rotation.from_quat(portion[:, [4, 5, 6, 3]])
        body = archive["body_pos"][first : last + 1]
        wrists = (
            body[:, names.index("right_wrist_yaw_link")]
            - body[:, names.index("pelvis")]
        )
        local_wrists = rotations.inv().apply(wrists)
        # torso orientation includes waist pitch, measured via FK on actual states.
        data = mujoco.MjData(model)
        torso_pitch = []
        for state in portion:
            data.qpos[:] = state
            mujoco.mj_forward(model, data)
            torso_rotation = Rotation.from_matrix(
                data.xmat[model.body("torso_link").id].reshape(3, 3)
            )
            torso_pitch.append(torso_rotation.as_euler("xyz")[1])
        segment_metrics.append(
            dict(
                name=segment["name"],
                interval_s=[first / 50, last / 50],
                actual_displacement_xy=(portion[-1, :2] - portion[0, :2]).tolist(),
                actual_path_m=float(
                    np.linalg.norm(np.diff(portion[:, :2], axis=0), axis=1).sum()
                ),
                yaw_change_deg=float(np.rad2deg(yaw[last] - yaw[first])),
                wrist_relative_range_xyz=np.ptp(local_wrists, axis=0).tolist(),
                wrist_relative_height_max=float(local_wrists[:, 2].max()),
                torso_pitch_deg=[
                    float(np.rad2deg(torso_pitch[0])),
                    float(np.rad2deg(max(torso_pitch))),
                    float(np.rad2deg(torso_pitch[-1])),
                ],
                support_min_N=float(telemetry[first:last, :2].sum(axis=1).min()),
                unsupported_seconds=float(telemetry[first:last, 8].sum() * 0.002),
            )
        )
    transition_metrics = []
    for transition in composition["transitions"]:
        first = round(transition["start"] * 50)
        last = min(len(states) - 1, round(transition["end"] * 50))
        if first >= last:
            continue
        foot = archive["body_pos"][
            first : last + 1,
            names.index(transition["planted_foot"] + "_ankle_roll_link"),
            :2,
        ]
        transition_metrics.append(
            dict(
                **transition,
                actual_plant_excursion_m=float(
                    np.linalg.norm(foot - foot[0], axis=1).max()
                ),
                support_min_N=float(telemetry[first:last, :2].sum(axis=1).min()),
                unsupported_seconds=float(telemetry[first:last, 8].sum() * 0.002),
                peak_joint_velocity_rad_s=float(
                    np.abs(archive["qvel"][first:last, 6:35]).max()
                ),
            )
        )
    limits = model.jnt_range[1:30]
    violation = np.maximum(
        limits[:, 0] - states[:, 7:36], states[:, 7:36] - limits[:, 1]
    )
    metrics = dict(
        complete=result["complete"],
        return_distance_m=return_distance,
        return_threshold_m=float(config.return_threshold_m),
        heading_error_deg=heading_error,
        heading_threshold_deg=float(config.heading_threshold_deg),
        return_pass=return_distance <= config.return_threshold_m,
        heading_pass=heading_error <= config.heading_threshold_deg,
        root_height_min_m=float(states[:, 2].min()),
        root_reference_rmse_m=float(
            np.sqrt(
                np.mean(
                    np.sum((states[:, :2] - reference[: len(states), :2]) ** 2, axis=1)
                )
            )
        ),
        joint_reference_rmse_rad=float(
            np.sqrt(np.mean((states[:, 7:36] - reference[: len(states), 7:36]) ** 2))
        ),
        max_joint_limit_violation_rad=float(max(0, violation.max())),
        max_torque_limit_ratio=float((archive["peak_actuator_force"] / EFFORT).max()),
        nonfoot_ground_contacts=int(telemetry[:, 5].sum()),
        unsupported_seconds=float(telemetry[:, 8].sum() * 0.002),
        max_penetration_m=float(-telemetry[:, 4].min()),
        mean_contact_slip_m_s=float(telemetry[:, 2].mean()),
        peak_contact_slip_m_s=float(telemetry[:, 3].max()),
        segments=segment_metrics,
        transitions=transition_metrics,
    )
    (destination / "metrics.json").write_text(json.dumps(metrics, indent=2))
    print(
        json.dumps(
            {
                name: value
                for name, value in metrics.items()
                if name not in ["segments", "transitions"]
            }
        ),
        flush=True,
    )


def run_phase(config):
    if config.phase == "compose":
        compose(config)
    elif config.phase == "audit":
        audit(config)
    elif config.phase == "verify":
        from g1_demo2_verify import verify

        verify(config)
    elif config.phase == "render":
        render(config)
    else:
        raise ValueError(config.phase)


def render(config):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont

    folder = Path(config.output) / "attempts" / config.attempt
    archive = np.load(folder / "actual.npz")
    composition = json.loads((folder / "composition.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=720, width=1280)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    lower = archive["qpos"][:, :2].min(axis=0)
    upper = archive["qpos"][:, :2].max(axis=0)
    camera.lookat[:] = [*(0.5 * (lower + upper)), 0.8]
    camera.distance = max(3.5, float(np.linalg.norm(upper - lower)) * 1.4 + 1.5)
    camera.azimuth = 125
    camera.elevation = -22
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 23)
    except OSError:
        font = ImageFont.load_default()
    samples = {
        "overview": [
            (f"Overview {index+1}", timestamp)
            for index, timestamp in enumerate(
                np.linspace(0, (len(archive["qpos"]) - 1) / 50, 15)
            )
        ],
        "transitions": [],
    }
    for index, transition in enumerate(composition["transitions"]):
        samples["transitions"].extend(
            [
                (f"Transition {index+1} {label}", timestamp)
                for label, timestamp in zip(
                    ["before", "middle", "after"],
                    [
                        transition["start"] - 0.12,
                        (transition["start"] + transition["end"]) / 2,
                        transition["end"] + 0.12,
                    ],
                )
            ]
        )
    selected_frames = {
        round(timestamp * 50) for group in samples.values() for _, timestamp in group
    }
    images = {}
    writer = (
        imageio.get_writer(
            folder / "continuous.mp4", fps=50, codec="libx264", quality=8
        )
        if config.get("video", True)
        else None
    )
    for frame_index, state in enumerate(archive["qpos"]):
        if writer is None and frame_index not in selected_frames:
            continue
        # Offline rendering of recorded physical states; no interaction with rollout.
        data.qpos[:] = state
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera)
        tile = Image.fromarray(renderer.render())
        stage = "FINAL HOLD"
        for segment in composition["segments"]:
            if segment["start"] <= frame_index / 50 <= segment["end"]:
                stage = segment["name"].upper()
        for transition in composition["transitions"]:
            if transition["start"] < frame_index / 50 < transition["end"]:
                stage = "SUPPORTED TRANSITION"
        ImageDraw.Draw(tile).text(
            (20, 20),
            f"ACTUAL CONTINUOUS PHYSICS | {stage} | {frame_index/50:.2f}s",
            font=font,
            fill="white",
            stroke_width=1,
            stroke_fill="black",
        )
        if writer:
            writer.append_data(np.array(tile))
        if frame_index in selected_frames:
            images[frame_index] = tile.resize((640, 360))
    if writer:
        writer.close()
    renderer.close()
    for name, group in samples.items():
        montage = Image.new("RGB", (1920, 360 * ((len(group) + 2) // 3)))
        metadata = []
        for index, (label, timestamp) in enumerate(group):
            frame_index = round(timestamp * 50)
            if frame_index not in images:
                continue
            montage.paste(images[frame_index], ((index % 3) * 640, (index // 3) * 360))
            metadata.append(
                dict(
                    label=label,
                    frame=frame_index,
                    time_s=frame_index / 50,
                    source="actual.npz:qpos",
                )
            )
        montage.save(folder / f"{name}.jpg", quality=95)
        (folder / f"{name}_frames.json").write_text(json.dumps(metadata, indent=2))
