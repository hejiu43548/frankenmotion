"""Independent artifact measurements, provenance, and repeat-array comparison."""

import ast
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform

import mujoco
import numpy as np
from omegaconf import OmegaConf
from scipy.signal import find_peaks
from scipy.spatial.transform import Rotation


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(config):
    root = Path(config.output)
    folder = root / "attempts" / config.attempt
    archive = np.load(folder / "actual.npz")
    composition = json.loads((folder / "composition.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
    data = mujoco.MjData(model)
    states = archive["qpos"]
    reference = archive["reference"]
    body_names = archive["body_names"].tolist()
    rotations = Rotation.from_quat(states[:, [4, 5, 6, 3]])
    torso_pitch = []
    for state in states:
        data.qpos[:] = state
        mujoco.mj_forward(model, data)
        torso_pitch.append(
            Rotation.from_matrix(
                data.xmat[model.body("torso_link").id].reshape(3, 3)
            ).as_euler("xyz")[1]
        )
    torso_pitch = np.rad2deg(torso_pitch)
    wave = next(
        segment for segment in composition["segments"] if segment["name"] == "wave"
    )
    first, last = [round(wave[name] * 50) for name in ["start", "end"]]
    wave_metrics = {}
    for side in ["left", "right"]:
        relative = (
            archive["body_pos"][:, body_names.index(side + "_wrist_yaw_link")]
            - archive["body_pos"][:, body_names.index("pelvis")]
        )
        local = rotations.inv().apply(relative)
        selected = local[first : last + 1]
        peaks, _ = find_peaks(selected[:, 1], prominence=0.035, distance=10)
        troughs, _ = find_peaks(-selected[:, 1], prominence=0.035, distance=10)
        wave_metrics[side] = dict(
            local_range_xyz_m=np.ptp(selected, axis=0).tolist(),
            local_height_max_m=float(selected[:, 2].max()),
            lateral_peaks_s=(peaks / 50 + first / 50).tolist(),
            lateral_troughs_s=(troughs / 50 + first / 50).tolist(),
        )
    boundary_metrics = []
    for transition in composition["transitions"]:
        for timestamp in [transition["start"], transition["end"]]:
            index = round(timestamp * 50)
            boundary_metrics.append(
                dict(
                    time_s=timestamp,
                    reference_joint_step_rad=float(
                        np.abs(
                            reference[index, 7:36] - reference[index - 1, 7:36]
                        ).max()
                    ),
                    actual_joint_step_rad=float(
                        np.abs(states[index, 7:36] - states[index - 1, 7:36]).max()
                    ),
                    actual_root_step_m=float(
                        np.linalg.norm(states[index, :3] - states[index - 1, :3])
                    ),
                    actual_velocity_step_rad_s=float(
                        np.abs(
                            archive["qvel"][index, 6:35]
                            - archive["qvel"][index - 1, 6:35]
                        ).max()
                    ),
                )
            )
    hold_first = round(composition["segments"][-1]["end"] * 50)
    telemetry = archive["telemetry"]
    hold = dict(
        start_s=hold_first / 50,
        seconds=(len(states) - 1 - hold_first) / 50,
        root_drift_m=float(np.linalg.norm(states[-1, :2] - states[hold_first, :2])),
        support_min_N=float(telemetry[hold_first:, :2].sum(axis=1).min()),
        unsupported_seconds=float(telemetry[hold_first:, 8].sum() * 0.002),
    )
    joint_errors = np.sqrt(np.mean((states[:, 7:36] - reference[:, 7:36]) ** 2, axis=0))
    assets = [
        Path(config.checkpoint),
        Path(config.robot_xml),
        Path(config.training_config),
        Path(config.manifest),
    ]
    assets += list(Path(config.policy).glob("*.onnx"))
    training = OmegaConf.load(config.training_config)
    assets += [Path(training.backbone.checkpoint), Path(training.data.skeleton)]
    assets += list(
        (Path(config.robot_xml).parent / "../meshes/g1").resolve().glob("*.STL")
    )
    assets += [
        Path(config.snapshot) / "work/GMR/general_motion_retargeting" / name
        for name in ["motion_retarget.py", "ik_configs/smplx_to_g1.json"]
    ]
    asset_hashes = {str(path): digest(path) for path in assets}
    assert asset_hashes[config.checkpoint] == config.checkpoint_sha256
    assert (
        asset_hashes[str(training.backbone.checkpoint)]
        == training.backbone.expected_sha256
    )
    prior = Path(
        "/mnt/sda2/frankenmotion/outputs_amass/g1_demo_1_20261010/attempts/demo1_final/controller_source_hashes.json"
    )
    controller_hashes = json.loads(prior.read_text())
    for path, value in controller_hashes.items():
        assert digest(path) == value
    (folder / "hashes.json").write_text(json.dumps(asset_hashes, indent=2))
    (folder / "controller_source_hashes.json").write_text(
        json.dumps(controller_hashes, indent=2)
    )
    source_checks = {}
    for path in Path(__file__).parent.glob("g1_demo*.py"):
        tree = ast.parse(path.read_text())
        short_names = sorted(
            {
                node.id
                for node in ast.walk(tree)
                if isinstance(node, ast.Name) and len(node.id) == 1 and node.id != "_"
            }
        )
        assert not short_names, (path, short_names)
        source_checks[path.name] = digest(path)
    layer_metrics = []
    for segment in config.plan:
        name = segment.source
        human = np.load(root / "generated" / f"{name}.npz")
        robot = np.load(root / "retarget" / f"{name}.npz")["qpos"]
        robot_positions = []
        for state in robot:
            data.qpos[:] = state
            mujoco.mj_forward(model, data)
            robot_positions.append(
                [
                    data.xpos[model.body(body).id] - data.xpos[model.body("pelvis").id]
                    for body in [
                        "left_ankle_roll_link",
                        "right_ankle_roll_link",
                        "left_wrist_yaw_link",
                        "right_wrist_yaw_link",
                    ]
                ]
            )
        human_relative = (
            (human["positions"] - human["positions"][:, :1])
            * 1.30
            / float(human["human_height"])
        )
        errors = np.linalg.norm(
            np.array(robot_positions) - human_relative[:, [7, 8, 20, 21]], axis=-1
        )
        layer_metrics.append(
            dict(
                name=segment.name,
                source=name,
                human_sha256=digest(root / "generated" / f"{name}.npz"),
                retarget_sha256=digest(root / "retarget" / f"{name}.npz"),
                pelvis_relative_fk_mean_error_m=errors.mean(axis=0).tolist(),
                body_order=["left_ankle", "right_ankle", "left_wrist", "right_wrist"],
                human_yaw_change_deg=float(np.rad2deg(human["motion"][:-1, 3].sum())),
                retarget_yaw_change_deg=float(
                    np.rad2deg(
                        np.unwrap(
                            Rotation.from_quat(robot[:, [4, 5, 6, 3]]).as_euler("xyz")[
                                :, 2
                            ]
                        )[-1]
                        - Rotation.from_quat(robot[0, [4, 5, 6, 3]]).as_euler("xyz")[2]
                    )
                ),
            )
        )
    ledger = []
    for path in sorted((root / "attempts").glob("*/metrics.json")):
        metrics = json.loads(path.read_text())
        ledger.append(
            dict(
                attempt=path.parent.name,
                complete=metrics["complete"],
                return_distance_m=metrics["return_distance_m"],
                heading_error_deg=metrics["heading_error_deg"],
                config=str(path.parent / "config.yaml"),
            )
        )
    report = dict(
        wave=wave_metrics,
        boundaries=boundary_metrics,
        final_hold=hold,
        torso_pitch_max_deg=float(torso_pitch.max()),
        joint_rmse_by_name={
            model.joint(index + 1).name: float(value)
            for index, value in enumerate(joint_errors)
        },
        reference_return_distance_m=float(
            np.linalg.norm(reference[-1, :2] - reference[0, :2])
        ),
        layers=layer_metrics,
        attempts=ledger,
        source_checks=source_checks,
        runtime=dict(
            python=platform.python_version(),
            generation_device="cpu",
            generation_batch_size=6,
            seeds=list(config.seeds),
            ddim_steps=50,
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
        ),
    )
    (folder / "verification.json").write_text(json.dumps(report, indent=2))
    np.savez_compressed(
        folder / "semantic_timeseries.npz",
        torso_pitch_deg=torso_pitch,
        time_s=np.arange(len(states)) / 50,
    )
    print(json.dumps(dict(wave=wave_metrics, final_hold=hold)), flush=True)


def compare(original_root, repeated_root, original_attempt, repeated_attempt):
    comparisons = {}
    for layer in ["generated", "retarget"]:
        comparisons[layer] = []
        for path in sorted((Path(original_root) / layer).glob("*.npz")):
            original = np.load(path)
            repeated = np.load(Path(repeated_root) / layer / path.name)
            differences = {
                name: float(np.max(np.abs(original[name] - repeated[name])))
                for name in original.files
            }
            assert all(value == 0 for value in differences.values()), (
                path,
                differences,
            )
            comparisons[layer].append(dict(file=path.name, max_errors=differences))
    original = np.load(
        Path(original_root) / "attempts" / original_attempt / "actual.npz"
    )
    repeated = np.load(
        Path(repeated_root) / "attempts" / repeated_attempt / "actual.npz"
    )
    comparisons["physics"] = {
        name: dict(
            equal=bool(np.array_equal(original[name], repeated[name])),
            max_error=float(np.max(np.abs(original[name] - repeated[name]))),
        )
        for name in [
            "qpos",
            "qvel",
            "body_pos",
            "reference",
            "peak_actuator_force",
            "telemetry",
        ]
    }
    assert all(row["equal"] for row in comparisons["physics"].values())
    return comparisons
