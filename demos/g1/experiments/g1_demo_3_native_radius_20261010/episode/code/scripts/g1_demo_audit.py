"""Read-only measurement of reference, actual states, contacts, and asset provenance."""

import hashlib
import json
from pathlib import Path

import mujoco
import numpy as np
from scipy.spatial.transform import Rotation
from omegaconf import OmegaConf
from g1_demo_physics import EFFORT


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def summarize_contacts(rows):
    if not rows:
        return dict(count=0)
    times = np.array([row["time"] for row in rows])
    forces = np.array([row["force"] for row in rows])
    return dict(
        count=len(rows),
        first_s=float(times.min()),
        last_s=float(times.max()),
        peak_force_N=float(forces.max()),
        normal_impulse_Ns=float(forces.sum() * 0.002),
        active_substeps=len(np.unique(np.round(times, 6))),
        maximum_penetration_m=float(max(0.0, -min(row["distance"] for row in rows))),
    )


def audit(config):
    folder = Path(config.output) / "attempts" / config.attempt
    archive = np.load(folder / "actual.npz")
    actual = archive["qpos"]
    reference = archive["reference"]
    telemetry = archive["telemetry"]
    result = json.loads((folder / "result.json").read_text())
    composition = json.loads((folder / "composition.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
    bodies = archive["body_names"].tolist()
    robot_rotation = Rotation.from_quat(actual[:, [4, 5, 6, 3]])
    tilt = np.arccos(np.clip(robot_rotation.as_matrix()[:, 2, 2], -1, 1))
    bag_id = model.body("bag").id
    bag_rotations = Rotation.from_quat(actual[:, [37, 38, 39, 36]])
    bag_centers = archive["body_pos"][:, bag_id] + bag_rotations.apply(
        model.body_ipos[bag_id]
    )
    bag_displacement = np.linalg.norm(bag_centers - bag_centers[0], axis=1)
    joint_error = actual[:, 7:36] - reference[: len(actual), 7:36]
    joint_ranges = model.jnt_range[1:30]
    violation = np.maximum(
        np.maximum(
            joint_ranges[:, 0] - actual[:, 7:36], actual[:, 7:36] - joint_ranges[:, 1]
        ),
        0.0,
    )
    contacts = [row for row in result["contacts"] if row["force"] > 1]
    categories = {
        "kick_foot": [
            row for row in contacts if row["body"] == "right_ankle_roll_link"
        ],
        "right_hand": [
            row for row in contacts if row["body"] == "right_wrist_yaw_link"
        ],
        "incidental_wrist_forearm": [
            row
            for row in contacts
            if row["body"] != "right_ankle_roll_link"
            and row["body"] != "right_wrist_yaw_link"
        ],
    }
    contact_metrics = {
        name: summarize_contacts(rows) for name, rows in categories.items()
    }
    # Separate disjoint hand impacts by >= 60 ms without a force-bearing contact.
    hand_events = []
    event = []
    for row in categories["right_hand"]:
        if event and row["time"] - event[-1]["time"] > 0.06:
            hand_events.append(summarize_contacts(event))
            event = []
        event.append(row)
    if event:
        hand_events.append(summarize_contacts(event))
    transition_metrics = []
    for transition in composition["transitions"]:
        start = round(transition["start"] * 50)
        end = round(transition["end"] * 50)
        values = dict(transition)
        actual_window = actual[start : end + 1]
        reference_window = reference[start : end + 1]
        foot = archive["body_pos"][
            start : end + 1, bodies.index("left_ankle_roll_link")
        ]
        values.update(
            reference_joint_step_max_rad=float(
                np.abs(np.diff(reference_window[:, 7:36], axis=0)).max()
            ),
            actual_joint_step_max_rad=float(
                np.abs(np.diff(actual_window[:, 7:36], axis=0)).max()
            ),
            actual_root_step_max_m=float(
                np.linalg.norm(np.diff(actual_window[:, :3], axis=0), axis=1).max()
            ),
            actual_joint_velocity_max_rad_s=float(
                np.abs(archive["qvel"][start : end + 1, 6:35]).max()
            ),
            actual_left_ankle_planar_excursion_m=float(
                np.linalg.norm(foot[:, :2] - foot[0, :2], axis=1).max()
            ),
            actual_min_support_N=float(telemetry[start:end, :2].sum(1).min()),
            actual_contact_slip_mean_m_s=float(telemetry[start:end, 2].mean()),
            actual_contact_slip_peak_m_s=float(telemetry[start:end, 3].max()),
        )
        boundary_metrics = []
        for boundary in [start, end]:
            boundary_metrics.append(
                dict(
                    time_s=boundary * 0.02,
                    reference_joint_step_rad=float(
                        np.abs(
                            reference[boundary, 7:36] - reference[boundary - 1, 7:36]
                        ).max()
                    ),
                    actual_joint_step_rad=float(
                        np.abs(
                            actual[boundary, 7:36] - actual[boundary - 1, 7:36]
                        ).max()
                    ),
                    actual_joint_velocity_change_rad_s=float(
                        np.abs(
                            archive["qvel"][boundary, 6:35]
                            - archive["qvel"][boundary - 1, 6:35]
                        ).max()
                    ),
                )
            )
        values["boundaries"] = boundary_metrics
        transition_metrics.append(values)
    segment_metrics = []
    for segment in composition["segments"]:
        start = round(segment["start"] * 50)
        end = min(len(actual) - 1, round(segment["end"] * 50))
        segment_metrics.append(
            dict(
                segment,
                actual_root_displacement_m=(
                    actual[end, :3] - actual[start, :3]
                ).tolist(),
                joint_tracking_rmse_rad=float(
                    np.sqrt(np.mean(joint_error[start : end + 1] ** 2))
                ),
                bag_com_max_displacement_m=float(
                    bag_displacement[start : end + 1].max()
                ),
            )
        )
    metrics = dict(
        continuous_physics=True,
        initialization_count=1,
        reset_at_transitions=False,
        physics_timestep_s=0.002,
        control_timestep_s=0.02,
        duration_s=result["duration"],
        complete=result["complete"],
        fall_time=result["fall_time"],
        pelvis_height_min_m=float(actual[:, 2].min()),
        pelvis_tilt_max_deg=float(np.rad2deg(tilt.max())),
        robot_nonfoot_ground_contact_count=int(telemetry[:, 5].sum()),
        maximum_joint_limit_violation_rad=float(violation.max()),
        maximum_actuator_force_over_limit_Nm=float(
            np.maximum(
                archive["peak_actuator_force"] - EFFORT,
                0.0,
            ).max()
        ),
        mean_torque_saturation_fraction=float(telemetry[:, 6].mean()),
        maximum_contact_penetration_m=float(-telemetry[:, 4].min()),
        support_contact_slip_mean_m_s=float(telemetry[:, 2].mean()),
        support_contact_slip_p95_m_s=float(np.quantile(telemetry[:, 2], 0.95)),
        support_contact_slip_peak_m_s=float(telemetry[:, 3].max()),
        no_support_control_frames=int(np.count_nonzero(telemetry[:, :2].sum(1) < 5)),
        joint_tracking_rmse_rad=float(np.sqrt(np.mean(joint_error**2))),
        root_planar_reference_rmse_m=float(
            np.sqrt(
                np.mean(
                    np.sum((actual[:, :2] - reference[: len(actual), :2]) ** 2, axis=1)
                )
            )
        ),
        root_planar_error_final_m=result["root_planar_error_final"],
        bag_mass_kg=float(model.body_mass[bag_id]),
        bag_inertia_kg_m2=model.body_inertia[bag_id].tolist(),
        bag_center_of_mass_peak_displacement_m=float(bag_displacement.max()),
        bag_swing_peak_deg=float(np.rad2deg(bag_rotations.magnitude().max())),
        contacts=contact_metrics,
        hand_contact_events=hand_events,
        transitions=transition_metrics,
        segments=segment_metrics,
        known_limitations=[
            "Released 2350-input shared20 tracker and compiled scene were absent on Betail; native frozen SONIC v1.1 ONNX (1751 encoder/994 decoder) was used with its checked observation and PD contract.",
            "SONIC tracks poses/orientation without absolute root-position feedback; actual planar path differs from composed reference.",
            "Seeds and geometry were screened in simulation; this is one demonstrated episode, not a general task-success benchmark.",
            "Support contact is constrained in the reference, but actual contact shows brief impact slip and finite solver penetration; see measured values.",
            "Wrist/forearm contact is measured separately from the distal hand; rejected development versions had incidental guard contact.",
            "Video is an uncut replay of saved physical states, not a kinematic-reference animation.",
        ],
    )
    if telemetry.shape[1] >= 9:
        metrics["unsupported_physics_steps"] = int(telemetry[:, 8].sum())
        metrics["unsupported_physics_steps_after_initial_0_1s"] = int(
            telemetry[5:, 8].sum()
        )
    metrics["acceptance"] = dict(
        walk_approach=np.linalg.norm(actual[round(5.94 * 50), :2] - actual[0, :2])
        > 1.0,
        foot_contact=contact_metrics["kick_foot"].get("peak_force_N", 0) > 10,
        hand_contact=contact_metrics["right_hand"].get("peak_force_N", 0) > 10,
        bag_response=bag_displacement.max() > 0.02,
        no_fall=result["complete"],
        support_each_control_frame=bool(np.all(telemetry[:, :2].sum(1) >= 5)),
        planted_foot_transition_excursion=all(
            values["actual_left_ankle_planar_excursion_m"] < 0.03
            for values in transition_metrics
        ),
        joint_limits=float(violation.max()) < 1e-5,
        no_nonfoot_ground_contact=telemetry[:, 5].sum() == 0,
        torque_limits=np.max(archive["peak_actuator_force"] - EFFORT) < 1e-6,
        no_reference_pose_jump=all(
            values["reference_joint_step_max_rad"] < 0.15
            for values in transition_metrics
        ),
    )
    metrics["acceptance"] = {
        name: bool(value) for name, value in metrics["acceptance"].items()
    }
    (folder / "metrics.json").write_text(json.dumps(metrics, indent=2))
    assets = [Path(config.checkpoint), Path(config.robot_xml)] + list(
        Path(config.policy).glob("*.onnx")
    )
    assets += [
        Path(config.snapshot)
        / "work/GMR/general_motion_retargeting/motion_retarget.py",
        Path(config.snapshot)
        / "work/GMR/general_motion_retargeting/ik_configs/smplx_to_g1.json",
    ]
    training_config = OmegaConf.load(config.training_config)
    assets.extend(
        [
            Path(config.training_config),
            Path(training_config.backbone.checkpoint),
            Path(training_config.data.skeleton),
        ]
    )
    asset_hashes = {str(path): sha256(path) for path in assets}
    assert asset_hashes[config.checkpoint] == config.checkpoint_sha256
    assert (
        asset_hashes[str(training_config.backbone.checkpoint)]
        == training_config.backbone.expected_sha256
    )
    mesh_root = (Path(config.robot_xml).parent / "../meshes/g1").resolve()
    asset_hashes.update(
        {str(path): sha256(path) for path in sorted(mesh_root.glob("*.STL"))}
    )
    asset_hashes.update(
        {
            str(path): sha256(path)
            for path in [
                folder / "scene.xml",
                folder / "reference.npz",
                folder / "actual.npz",
                folder / "config.yaml",
            ]
        }
    )
    (folder / "hashes.json").write_text(json.dumps(asset_hashes, indent=2))
    print(
        json.dumps(
            {
                name: value
                for name, value in metrics.items()
                if name not in ["known_limitations", "transitions", "segments"]
            },
            indent=2,
        ),
        flush=True,
    )


def verify(config):
    import ast
    import importlib.metadata
    import platform

    root = Path(config.output)
    folder = root / "attempts" / config.attempt
    repeated = root / "attempts/demo1_verification"
    checks = {}
    original_states = np.load(folder / "actual.npz")
    repeated_states = np.load(repeated / "actual.npz")
    for name in [
        "qpos",
        "qvel",
        "body_pos",
        "reference",
        "peak_actuator_force",
        "telemetry",
    ]:
        difference = float(np.abs(original_states[name] - repeated_states[name]).max())
        checks[name] = dict(
            max_absolute_difference=difference,
            array_equal=bool(
                np.array_equal(original_states[name], repeated_states[name])
            ),
        )
        assert difference == 0.0, (name, difference)
    retarget_checks = []
    for path in sorted((root / "retarget").glob("*.npz")):
        original = np.load(path)["qpos"]
        repeated_reference = np.load(
            root / "verification_generation_cpu/retarget" / path.name
        )["qpos"]
        error = float(np.abs(original - repeated_reference).max())
        assert error == 0.0, (path.name, error)
        retarget_checks.append(dict(file=path.name, max_absolute_difference=error))
    generation = []
    for path in sorted((root / "generated").glob("*.npz")):
        original = np.load(path)
        repeated_motion = np.load(
            root / "verification_generation_cpu/generated" / path.name
        )
        error = max(
            float(np.abs(original[name] - repeated_motion[name]).max())
            for name in ["motion", "positions", "quaternions"]
        )
        assert error == 0.0, (path.name, error)
        generation.append(
            dict(
                file=path.name,
                max_absolute_difference=error,
                source_sha256=sha256(path),
            )
        )
    # Retarget error is a morphology-dependent pelvis-relative FK measure, not
    # evidence that robot joint coordinates equal human joint coordinates.
    model = mujoco.MjModel.from_xml_path(str(root / "retarget_model.xml"))
    data = mujoco.MjData(model)
    records = json.loads((root / "generated/manifest.json").read_text())
    layer_metrics = []
    for name in config.selection:
        original = np.load(root / "generated" / f"{name}.npz")
        states = np.load(root / "retarget" / f"{name}.npz")["qpos"]
        human_positions = original["positions"]
        human_positions = (
            (human_positions - human_positions[:, :1])
            * 1.30
            / float(original["human_height"])
        )
        robot_positions = []
        for state in states:
            data.qpos[:] = state
            mujoco.mj_forward(model, data)
            robot_positions.append(
                np.stack(
                    [
                        data.xpos[model.body(body).id]
                        - data.xpos[model.body("pelvis").id]
                        for body in [
                            "left_toe_link",
                            "right_toe_link",
                            "left_wrist_yaw_link",
                            "right_wrist_yaw_link",
                        ]
                    ]
                )
            )
        error = np.linalg.norm(
            np.array(robot_positions) - human_positions[:, [10, 11, 20, 21]], axis=-1
        )
        record = next(record for record in records if Path(record["file"]).stem == name)
        layer_metrics.append(
            dict(
                name=name,
                human=record,
                retarget_file=str(root / "retarget" / f"{name}.npz"),
                retarget_sha256=sha256(root / "retarget" / f"{name}.npz"),
                pelvis_relative_FK_error_mean_m=error.mean(0).tolist(),
                pelvis_relative_FK_error_body_order=[
                    "left_toe",
                    "right_toe",
                    "left_wrist",
                    "right_wrist",
                ],
                retarget_root_displacement_m=(states[-1, :3] - states[0, :3]).tolist(),
                source_frames=len(states),
                source_fps=20.0,
            )
        )
    ledger = []
    for path in sorted((root / "attempts").glob("*/result.json")):
        result = json.loads(path.read_text())
        contacts = [row for row in result["contacts"] if row["force"] > 1]
        arrays = np.load(path.parent / "actual.npz")
        telemetry = arrays["telemetry"] if "telemetry" in arrays else None
        ledger.append(
            dict(
                attempt=path.parent.name,
                complete=result["complete"],
                duration_s=result["duration"],
                foot_contact=any(
                    row["body"] == "right_ankle_roll_link" for row in contacts
                ),
                hand_contact=any(
                    row["body"] == "right_wrist_yaw_link" for row in contacts
                ),
                unsupported_control_frames=(
                    int(np.count_nonzero(telemetry[:, :2].sum(1) < 5))
                    if telemetry is not None
                    else None
                ),
                result=str(path),
                config=str(path.parent / "config.yaml"),
            )
        )
    source_folder = Path(__file__).parent
    syntax_checks = {}
    for path in sorted(source_folder.glob("g1_demo*.py")):
        tree = ast.parse(path.read_text())
        single_letter_names = sorted(
            {
                node.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Name) and len(node.id) == 1 and node.id != "_"
            }
        )
        assert not single_letter_names, (path, single_letter_names)
        syntax_checks[path.name] = dict(
            sha256=sha256(path), single_letter_names=single_letter_names
        )
    runtime = dict(
        python=platform.python_version(),
        generation_device="cpu",
        physics_device="cpu",
        rendering="EGL",
        packages={
            name: importlib.metadata.version(name)
            for name in [
                "torch",
                "numpy",
                "scipy",
                "mujoco",
                "mink",
                "daqp",
                "onnxruntime",
                "imageio",
                "imageio-ffmpeg",
                "Pillow",
            ]
        },
    )
    report = dict(
        physics_reproduction=checks,
        generation_reproduction=generation,
        retarget_reproduction=retarget_checks,
        layers=layer_metrics,
        attempts=ledger,
        source_checks=syntax_checks,
        runtime=runtime,
        sampling_warning="CPU and CUDA use distinct random streams for identical integer seeds. Initial auto-device verification differed; explicit generation_device=cpu reproduces all 18 original arrays exactly.",
    )
    (folder / "verification.json").write_text(json.dumps(report, indent=2))
    source_contract = (
        Path(config.robot_xml).parents[5]
        / "gear_sonic_deploy/src/g1/g1_deploy_onnx_ref"
    )
    provenance = {
        str(path): sha256(path)
        for path in [
            source_contract / "src/g1_deploy_onnx_ref.cpp",
            source_contract / "include/policy_parameters.hpp",
        ]
        if path.exists()
    }
    (folder / "controller_source_hashes.json").write_text(
        json.dumps(provenance, indent=2)
    )
    print(
        json.dumps(
            dict(
                physical_arrays=checks,
                generated_motions_exact=len(generation),
                runtime=runtime,
                source_checks=syntax_checks,
            ),
            indent=2,
        )
    )
