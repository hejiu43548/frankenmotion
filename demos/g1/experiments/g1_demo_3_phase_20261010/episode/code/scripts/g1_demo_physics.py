"""GMR conversion and physical SONIC rollout; qpos is written only at reset."""

import contextlib
import io
import json
import os
from pathlib import Path
import sys
import types
import xml.etree.ElementTree as ET

os.environ.setdefault("MUJOCO_GL", "egl")
import mujoco
import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation, Slerp

JOINT_NAMES = [
    "pelvis",
    "left_hip",
    "right_hip",
    "spine1",
    "left_knee",
    "right_knee",
    "spine2",
    "left_ankle",
    "right_ankle",
    "spine3",
    "left_foot",
    "right_foot",
    "neck",
    "left_collar",
    "right_collar",
    "head",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
]
DEFAULT = np.array(
    [-0.312, 0, 0, 0.669, -0.363, 0] * 2
    + [0, 0, 0]
    + [0.2, 0.2, 0, 0.6, 0, 0, 0]
    + [0.2, -0.2, 0, 0.6, 0, 0, 0]
)
ARMATURE = np.array(
    [0.025101925, 0.025101925, 0.010177520, 0.025101925, 0.003609725, 0.003609725] * 2
    + [0.010177520, 0.003609725, 0.003609725]
    + [0.003609725] * 5
    + [0.00425] * 2
    + [0.003609725] * 5
    + [0.00425] * 2
)
EFFORT = np.array(
    [139, 139, 88, 139, 25, 25] * 2
    + [88, 25, 25]
    + [25] * 5
    + [5] * 2
    + [25] * 5
    + [5] * 2
)
PROPORTIONAL = ARMATURE * (20 * np.pi) ** 2
DERIVATIVE = 4 * ARMATURE * 20 * np.pi
ACTION_SCALE = 0.25 * EFFORT / PROPORTIONAL
PROPORTIONAL[[4, 5, 10, 11, 13, 14]] *= 2
DERIVATIVE[[4, 5, 10, 11, 13, 14]] *= 2
MJ_TO_POLICY = np.array(
    [
        0,
        6,
        12,
        1,
        7,
        13,
        2,
        8,
        14,
        3,
        9,
        15,
        22,
        4,
        10,
        16,
        23,
        5,
        11,
        17,
        24,
        18,
        25,
        19,
        26,
        20,
        27,
        21,
        28,
    ]
)


def floor_height(model, data):
    mujoco.mj_forward(model, data)
    heights = []
    for geom_index in range(model.ngeom):
        name = model.body(model.geom_bodyid[geom_index]).name
        if "ankle_roll" not in name or not (
            model.geom_contype[geom_index] or model.geom_conaffinity[geom_index]
        ):
            continue
        rotation = data.geom_xmat[geom_index].reshape(3, 3)
        position = data.geom_xpos[geom_index]
        size = model.geom_size[geom_index]
        if model.geom_type[geom_index] == mujoco.mjtGeom.mjGEOM_SPHERE:
            heights.append(position[2] - size[0])
        elif model.geom_type[geom_index] == mujoco.mjtGeom.mjGEOM_BOX:
            heights.append(position[2] - np.abs(rotation[2]).dot(size))
        elif model.geom_type[geom_index] == mujoco.mjtGeom.mjGEOM_MESH:
            mesh_id = model.geom_dataid[geom_index]
            vertices = model.mesh_vert[
                model.mesh_vertadr[mesh_id] : model.mesh_vertadr[mesh_id]
                + model.mesh_vertnum[mesh_id]
            ]
            heights.append((vertices @ rotation.T + position)[:, 2].min())
    return min(heights)


def scene(config, bag=None):
    root = ET.parse(config.robot_xml).getroot()
    for existing in list(root.findall("visual")) + list(root.findall("option")):
        root.remove(existing)
    for worldbody in root.findall("worldbody"):
        for geom in list(worldbody.findall("geom")):
            if geom.get("name") == "floor":
                worldbody.remove(geom)
    for side, lateral in [("left", ".003"), ("right", "-.003")]:
        wrist = root.find(f".//body[@name='{side}_wrist_yaw_link']")
        ET.SubElement(
            wrist,
            "geom",
            name=f"{side}_palm_collision",
            type="mesh",
            mesh=f"{side}_rubber_hand",
            pos=f".0415 {lateral} 0",
            mass="0",
            rgba=".65 .65 .65 1",
        )
    compiler = root.find("compiler")
    compiler.set(
        "meshdir",
        str((Path(config.robot_xml).parent / compiler.get("meshdir")).resolve()),
    )
    ET.SubElement(
        root, "option", timestep="0.002", integrator="implicitfast", iterations="60"
    )
    visual = ET.SubElement(root, "visual")
    ET.SubElement(visual, "global", offwidth="1280", offheight="720")
    ET.SubElement(visual, "headlight", ambient="0.5 0.5 0.5", diffuse="0.7 0.7 0.7")
    world = root.find("worldbody")
    ET.SubElement(
        world,
        "geom",
        name="floor",
        type="plane",
        size="10 10 .1",
        rgba=".85 .85 .85 1",
        material="groundplane",
        friction=f"{config.get('floor_friction', .9)} .02 .002",
        priority="1",
        solref=f"{config.get('floor_solref', .02)} 1",
        solimp=".95 .99 .001",
    )
    ET.SubElement(world, "light", pos="1 -2 4", dir="0 0 -1", diffuse=".9 .9 .9")
    for texture in root.findall(".//texture"):
        if texture.get("type") == "skybox":
            texture.set("builtin", "gradient")
            texture.set("rgb1", ".32 .40 .48")
            texture.set("rgb2", ".08 .12 .18")
    if bag is not None:
        body = ET.SubElement(world, "body", name="bag", pos=f"{bag[0]} {bag[1]} 1.75")
        ET.SubElement(
            body, "joint", name="bag_swing", type="ball", damping="2.5", stiffness="18"
        )
        ET.SubElement(
            body,
            "geom",
            name="bag_collision",
            type="capsule",
            fromto="0 0 -1.4 0 0 -.45",
            size=".15",
            mass="7",
            rgba=".75 .18 .12 1",
            friction=".7 .01 .001",
            solref=".012 1",
            solimp=".9 .95 .002",
        )
        ET.SubElement(
            body,
            "geom",
            name="bag_rope",
            type="capsule",
            fromto="0 0 0 0 0 -.3",
            size=".014",
            mass=".05",
            contype="0",
            conaffinity="0",
            rgba=".12 .12 .12 1",
        )
        ET.SubElement(
            world,
            "geom",
            name="stand_post",
            type="capsule",
            fromto=f"{bag[0]+.65} {bag[1]} .04 {bag[0]+.65} {bag[1]} 2.1",
            size=".04",
            rgba=".25 .28 .3 1",
        )
        ET.SubElement(
            world,
            "geom",
            name="stand_beam",
            type="capsule",
            fromto=f"{bag[0]+.65} {bag[1]} 2.1 {bag[0]} {bag[1]} 2.1",
            size=".04",
            rgba=".25 .28 .3 1",
        )
        ET.SubElement(
            world,
            "geom",
            name="stand_base",
            type="box",
            pos=f"{bag[0]+.65} {bag[1]} .025",
            size=".35 .45 .025",
            rgba=".25 .28 .3 1",
        )
        ET.SubElement(
            world,
            "geom",
            name="support",
            type="capsule",
            fromto=f"{bag[0]} {bag[1]} 1.8 {bag[0]} {bag[1]} 2.1",
            size=".04",
            contype="0",
            conaffinity="0",
            rgba=".3 .3 .3 1",
        )
    model = mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"))
    model.dof_armature[6:35] = ARMATURE
    return model, ET.tostring(root, encoding="unicode")


def retarget(config):
    package = types.ModuleType("general_motion_retargeting")
    package.__path__ = [
        str(Path(config.snapshot) / "work/GMR/general_motion_retargeting")
    ]
    sys.modules[package.__name__] = package
    from general_motion_retargeting import params

    marker_root = ET.parse(config.robot_xml).getroot()
    marker_root.find("compiler").set(
        "meshdir",
        str(
            (
                Path(config.robot_xml).parent
                / marker_root.find("compiler").get("meshdir")
            ).resolve()
        ),
    )
    for side in ["left", "right"]:
        ankle = marker_root.find(f".//body[@name='{side}_ankle_roll_link']")
        ET.SubElement(ankle, "body", name=f"{side}_toe_link", pos="0.1 0 -0.02")
    marker_path = Path(config.output) / "retarget_model.xml"
    marker_path.write_text(ET.tostring(marker_root, encoding="unicode"))
    params.ROBOT_XML_DICT["unitree_g1"] = marker_path
    from general_motion_retargeting.motion_retarget import GeneralMotionRetargeting

    destination = Path(config.output) / "retarget"
    destination.mkdir(exist_ok=True)
    model, _ = scene(config)
    data = mujoco.MjData(model)
    records = json.loads((Path(config.output) / "generated/manifest.json").read_text())
    for record in records:
        target = destination / Path(record["file"]).name
        if target.exists():
            continue
        archive = np.load(record["file"])
        with contextlib.redirect_stdout(io.StringIO()):
            solver = GeneralMotionRetargeting("smplx", "unitree_g1", verbose=False)
        solver.human_scale_table = {
            name: 1.30 / float(archive["human_height"])
            for name in solver.human_scale_table
        }
        states = []
        for positions, quaternions in zip(archive["positions"], archive["quaternions"]):
            frame = {
                name: (positions[joint_index], quaternions[joint_index])
                for joint_index, name in enumerate(JOINT_NAMES)
            }
            states.append(solver.retarget(frame, offset_to_ground=False).copy())
        states = np.array(states)
        # A single constant floor correction preserves the generated root trajectory.
        data.qpos[:] = states[0]
        states[:, 2] -= floor_height(model, data)
        np.savez_compressed(target, qpos=states, fps=20.0)
        print(
            "retarget",
            target.name,
            "height",
            states[:, 2].min(),
            states[:, 2].max(),
            flush=True,
        )


class SonicPolicy:
    def __init__(self, folder):
        import onnxruntime
        from collections import deque

        options = onnxruntime.SessionOptions()
        options.intra_op_num_threads = 2
        options.inter_op_num_threads = 1
        self.encoder = onnxruntime.InferenceSession(
            str(Path(folder) / "model_encoder.onnx"),
            options,
            providers=["CPUExecutionProvider"],
        )
        self.decoder = onnxruntime.InferenceSession(
            str(Path(folder) / "model_decoder.onnx"),
            options,
            providers=["CPUExecutionProvider"],
        )
        assert self.encoder.get_inputs()[0].shape == [1, 1751]
        assert self.decoder.get_inputs()[0].shape == [1, 994]
        self.deque = deque
        self.history = None

    def state(self, data, previous):
        rotation = Rotation.from_quat(data.qpos[[4, 5, 6, 3]])
        return [
            data.qvel[3:6].copy(),
            (data.qpos[7:36] - DEFAULT)[MJ_TO_POLICY],
            data.qvel[6:35][MJ_TO_POLICY],
            previous.copy(),
            rotation.inv().apply([0.0, 0.0, -1.0]),
        ]

    def update(self, data, previous):
        state = self.state(data, previous)
        if self.history is None:
            self.history = [
                self.deque([value.copy() for _ in range(10)], maxlen=10)
                for value in state
            ]
        else:
            for history, value in zip(self.history, state):
                history.append(value.copy())

    def action(self, data, reference, velocity, frame_index):
        indices = np.minimum(frame_index + np.arange(10) * 5, len(reference) - 1)
        robot_rotation = Rotation.from_quat(data.qpos[[4, 5, 6, 3]])
        reference_rotations = Rotation.from_quat(reference[indices][:, [4, 5, 6, 3]])
        robot_rotation = Rotation.from_euler("z", robot_rotation.as_euler("xyz")[2])
        observation = np.zeros(1751, np.float32)
        observation[4:294] = reference[indices, 7:36][:, MJ_TO_POLICY].ravel()
        observation[294:584] = velocity[indices][:, MJ_TO_POLICY].ravel()
        observation[584:644] = (
            (robot_rotation.inv() * reference_rotations).as_matrix()[:, :, :2].ravel()
        )
        latent = self.encoder.run(None, {"obs_dict": observation[None]})[0].ravel()
        decoder_input = np.concatenate(
            [latent] + [np.array(history).ravel() for history in self.history]
        ).astype(np.float32)
        raw = np.clip(
            self.decoder.run(None, {"obs_dict": decoder_input[None]})[0].ravel(),
            -20,
            20,
        )
        assert raw.shape == (29,) and np.isfinite(raw).all()
        return DEFAULT + raw[np.argsort(MJ_TO_POLICY)] * ACTION_SCALE, raw


def resample(states, speed=1.0):
    times = np.arange(len(states)) / 20 / speed
    samples = np.arange(int(times[-1] / 0.02) + 1) * 0.02
    result = np.stack([np.interp(samples, times, column) for column in states.T], 1)
    result[:, 3:7] = Slerp(times, Rotation.from_quat(states[:, [4, 5, 6, 3]]))(
        samples
    ).as_quat()[:, [3, 0, 1, 2]]
    return result


def simulate(config, reference, destination, bag=None):
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    if (destination / "actual.npz").exists():
        raise FileExistsError(f"Preserving previous episode: {destination}")
    model, xml = scene(config, bag)
    (destination / "scene.xml").write_text(xml)
    data = mujoco.MjData(model)
    data.qpos[:36] = reference[0]
    data.qpos[2] -= floor_height(model, data)
    mujoco.mj_forward(model, data)
    assert model.nu == 29 and model.nq == (40 if bag is not None else 36)
    assert reference.shape[1] == 36 and np.isfinite(reference).all()
    policy = SonicPolicy(config.policy)
    policy.update(data, np.zeros(29))
    velocity = np.gradient(reference[:, 7:36], 0.02, axis=0)
    states = [data.qpos.copy()]
    velocities = [data.qvel.copy()]
    body_positions = [data.xpos.copy()]
    forces = []
    contacts = []
    telemetry = []
    failure = None
    for frame_index in range(len(reference) - 1):
        target, action = policy.action(data, reference, velocity, frame_index)
        peak_force = np.zeros(29)
        support_forces = np.zeros(2)
        support_slip = []
        minimum_distance = 0.0
        nonfoot_ground = 0
        saturation_count = 0
        substep_support = []
        for _ in range(10):
            current_support = 0.0
            requested_torque = (
                PROPORTIONAL * (target - data.qpos[7:36]) - DERIVATIVE * data.qvel[6:35]
            )
            saturation_count += int(np.count_nonzero(np.abs(requested_torque) > EFFORT))
            data.ctrl[:29] = np.clip(requested_torque, -EFFORT, EFFORT)
            mujoco.mj_step(model, data)
            peak_force = np.maximum(peak_force, np.abs(data.actuator_force[:29]))
            for contact_index, contact in enumerate(data.contact):
                first = model.geom(contact.geom1)
                second = model.geom(contact.geom2)
                force = np.zeros(6)
                mujoco.mj_contactForce(model, data, contact_index, force)
                minimum_distance = min(minimum_distance, float(contact.dist))
                if first.name == "floor" or second.name == "floor":
                    other_id = contact.geom2 if first.name == "floor" else contact.geom1
                    body_id = int(model.geom_bodyid[other_id])
                    body_name = model.body(body_id).name
                    if "ankle_roll" in body_name:
                        side_index = 0 if "left" in body_name else 1
                        support_forces[side_index] += force[0] / 10
                        current_support += force[0]
                        if force[0] > 20:
                            jacobian = np.zeros((3, model.nv))
                            angular_jacobian = np.zeros((3, model.nv))
                            mujoco.mj_jac(
                                model,
                                data,
                                jacobian,
                                angular_jacobian,
                                contact.pos,
                                body_id,
                            )
                            support_slip.append(
                                float(np.linalg.norm((jacobian @ data.qvel)[:2]))
                            )
                    else:
                        nonfoot_ground += 1
                if first.name == "bag_collision" or second.name == "bag_collision":
                    other = (
                        contact.geom2
                        if first.name == "bag_collision"
                        else contact.geom1
                    )
                    contacts.append(
                        dict(
                            time=float(data.time),
                            body=model.body(model.geom_bodyid[other]).name,
                            force=float(force[0]),
                            distance=float(contact.dist),
                        )
                    )
            substep_support.append(current_support)
        mujoco.mj_forward(model, data)
        policy.update(data, action)
        telemetry.append(
            [
                *support_forces,
                np.mean(support_slip) if support_slip else 0.0,
                max(support_slip, default=0.0),
                minimum_distance,
                nonfoot_ground,
                saturation_count / 290,
                min(substep_support),
                sum(value < 5 for value in substep_support),
            ]
        )
        states.append(data.qpos.copy())
        velocities.append(data.qvel.copy())
        body_positions.append(data.xpos.copy())
        forces.append(peak_force)
        tilt = np.arccos(
            np.clip(data.xmat[model.body("pelvis").id].reshape(3, 3)[2, 2], -1, 1)
        )
        if not np.isfinite(data.qpos).all() or data.qpos[2] < 0.35 or tilt > np.pi / 3:
            failure = float(data.time)
            break
    states = np.array(states)
    np.savez_compressed(
        destination / "actual.npz",
        qpos=states,
        qvel=velocities,
        body_pos=body_positions,
        peak_actuator_force=forces,
        telemetry=telemetry,
        telemetry_names=np.array(
            [
                "left_support_N",
                "right_support_N",
                "mean_contact_slip_m_s",
                "max_contact_slip_m_s",
                "minimum_contact_distance_m",
                "nonfoot_ground_contacts",
                "torque_saturation_fraction",
                "minimum_substep_support_N",
                "unsupported_substeps",
            ]
        ),
        reference=reference,
        fps=50.0,
        body_names=np.array([model.body(index).name for index in range(model.nbody)]),
    )
    result = dict(
        complete=failure is None,
        fall_time=failure,
        frames=len(states),
        duration=(len(states) - 1) * 0.02,
        root_height_min=float(states[:, 2].min()),
        root_end=states[-1, :3].tolist(),
        root_planar_error_final=float(
            np.linalg.norm(states[-1, :2] - reference[len(states) - 1, :2])
        ),
        contact_count=len(contacts),
        contacts=contacts,
        state_resets=1,
        qpos_overwrites_during_rollout=0,
        controller="official SONIC ONNX; frozen encoder/decoder; 1751/994 obs; native torque PD",
    )
    (destination / "result.json").write_text(json.dumps(result, indent=2))
    print(
        destination.name,
        {name: value for name, value in result.items() if name != "contacts"},
        flush=True,
    )
    return result


def run_phase(config):
    if config.phase == "retarget":
        retarget(config)
    elif config.phase == "probe":
        for path in sorted((Path(config.output) / "retarget").glob("*.npz")):
            destination = Path(config.output) / "probes" / path.stem
            if (destination / "result.json").exists():
                continue
            reference = resample(np.load(path)["qpos"])
            simulate(config, reference, destination)
    elif config.phase == "compose":
        compose(config)
    elif config.phase == "verify":
        from g1_demo_audit import verify

        verify(config)
    elif config.phase == "audit":
        from g1_demo_audit import audit

        audit(config)
    elif config.phase == "render":
        render(Path(config.output) / config.render_dir, bool(config.video))
    else:
        raise ValueError(config.phase)


def render(folder, video=True):
    import imageio.v2 as imageio
    from PIL import Image, ImageDraw, ImageFont

    folder = Path(folder)
    archive = np.load(folder / "actual.npz")
    model = mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=720, width=1280)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [1.5, 0.2, 0.9]
    camera.distance = 3.7
    camera.azimuth = 110
    camera.elevation = -12
    composition = (
        json.loads((folder / "composition.json").read_text())
        if (folder / "composition.json").exists()
        else None
    )
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 23)
    except OSError:
        font = ImageFont.load_default()
    indices = np.linspace(0, len(archive["qpos"]) - 1, 8).astype(int)
    tiles = []
    writer = (
        imageio.get_writer(
            folder / "continuous.mp4", fps=50, codec="libx264", quality=8
        )
        if video
        else None
    )
    for frame_index, state in enumerate(archive["qpos"]):
        if not video and frame_index not in indices:
            continue
        data.qpos[:] = state
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera)
        pixels = renderer.render()
        tile = Image.fromarray(pixels)
        draw = ImageDraw.Draw(tile)
        stage = "SETTLE"
        if composition:
            for segment in composition["segments"]:
                if segment["start"] <= frame_index * 0.02 <= segment["end"]:
                    stage = segment["name"].split("_")[0].upper()
                    if (
                        stage == "KICK"
                        and composition.get("kick_recovery", False)
                        and frame_index * 0.02 > (segment["start"] + segment["end"]) / 2
                    ):
                        stage = "KICK RECOVERY / FOOT LANDING"
            for transition in composition["transitions"]:
                if transition["start"] < frame_index * 0.02 < transition["end"]:
                    stage = "CONTACT-CONSTRAINED TRANSITION"
        draw.text(
            (20, 20),
            f"CONTINUOUS PHYSICS | {stage} | {frame_index*.02:.2f}s",
            font=font,
            fill="white",
            stroke_width=1,
            stroke_fill="black",
        )
        if writer:
            writer.append_data(np.array(tile))
        if frame_index in indices:
            tiles.append(tile.resize((480, 270)))
    if writer:
        writer.close()
    renderer.close()
    montage = Image.new("RGB", (480 * 4, 270 * 2))
    for tile_index, tile in enumerate(tiles):
        montage.paste(tile, ((tile_index % 4) * 480, (tile_index // 4) * 270))
    montage.save(folder / "frames.jpg")
    if composition:
        render_keyframes(folder)


def place(states, yaw, translation):
    result = states.copy()
    rotation = Rotation.from_euler("z", yaw)
    result[:, :3] = rotation.apply(result[:, :3]) + np.asarray(translation)
    result[:, 3:7] = (rotation * Rotation.from_quat(result[:, [4, 5, 6, 3]])).as_quat()[
        :, [3, 0, 1, 2]
    ]
    return result


def endpoint_ease(states, seconds=0.3):
    """Ease time at both ends, retaining the complete source trajectory."""
    frame_count = len(states)
    grid = np.arange(frame_count, dtype=float)
    width = seconds / 0.02
    for frame_index in range(frame_count):
        if grid[frame_index] < width:
            fraction = grid[frame_index] / width
            grid[frame_index] = width * (2 * fraction**2 - fraction**3)
        elif grid[frame_index] > frame_count - 1 - width:
            fraction = (frame_count - 1 - grid[frame_index]) / width
            grid[frame_index] = (
                frame_count - 1 - width * (2 * fraction**2 - fraction**3)
            )
    result = np.stack(
        [np.interp(grid, np.arange(frame_count), column) for column in states.T], 1
    )
    result[:, 3:7] = Slerp(
        np.arange(frame_count), Rotation.from_quat(states[:, [4, 5, 6, 3]])
    )(grid).as_quat()[:, [3, 0, 1, 2]]
    return result


def compose(config):
    from scipy.optimize import least_squares

    model, _ = scene(config)
    data = mujoco.MjData(model)
    foot_ids = [model.body(f"{side}_ankle_roll_link").id for side in ["left", "right"]]

    def feet(state):
        data.qpos[:] = state
        mujoco.mj_forward(model, data)
        return data.xpos[foot_ids].copy()

    names = list(config.selection)
    sequences = []
    for name in names:
        states = np.load(Path(config.output) / "retarget" / f"{name}.npz")["qpos"]
        if config.get("source_ranges") is not None:
            interval = config.source_ranges[name.split("_")[0]]
            states = states[round(interval[0] * 20) : round(interval[1] * 20) + 1]
        speed = config.get("kick_speed", 1.0) if name.startswith("kick") else 1.0
        states = endpoint_ease(resample(states, speed=speed))
        yaw = Rotation.from_quat(states[0, [4, 5, 6, 3]]).as_euler("xyz")[2]
        states = place(states, -yaw, [0, 0, 0])
        states[:, :2] -= states[0, :2].copy()
        if name.startswith("kick") and config.get("kick_hold_seconds", 0):
            states = np.concatenate(
                [
                    states,
                    np.repeat(
                        states[-1:], round(config.kick_hold_seconds / 0.02), axis=0
                    ),
                ]
            )
        if name.startswith("kick") and config.get("kick_recovery", False):
            # Reuse the generated approach in reverse to put the kicking foot down
            # before changing to the boxing stance; no new motion model is used.
            states = np.concatenate([states, states[-2::-1]])
        sequences.append(states)
    result = sequences[0]
    segments = [dict(name=names[0], start=0.0, end=(len(result) - 1) * 0.02)]
    transitions = []
    for name, states in zip(names[1:], sequences[1:]):
        previous = result[-1]
        previous_yaw = Rotation.from_quat(previous[[4, 5, 6, 3]]).as_euler("xyz")[2]
        states = place(states, previous_yaw, [0, 0, 0])
        # Align the planted left ankle, including floor height, before bridging.
        shift = feet(previous)[0] - feet(states[0])[0]
        states[:, :3] += shift
        start_feet = feet(previous)
        end_feet = feet(states[0])
        frame_count = round(config.transition_seconds / 0.02)
        fractions = np.linspace(0, 1, frame_count + 1)
        smooth = fractions**3 * (10 - 15 * fractions + 6 * fractions**2)
        bridge = previous[None] * (1 - smooth[:, None]) + states[:1] * smooth[:, None]
        bridge[:, 3:7] = Slerp(
            [0, 1], Rotation.from_quat(np.stack([previous, states[0]])[:, [4, 5, 6, 3]])
        )(smooth).as_quat()[:, [3, 0, 1, 2]]
        indices = np.r_[np.arange(3), np.arange(7, 19)]
        lower = np.r_[[-np.inf] * 3, model.jnt_range[1:13, 0]]
        upper = np.r_[[np.inf] * 3, model.jnt_range[1:13, 1]]
        errors = []
        for frame_index in range(1, frame_count):
            desired = bridge[frame_index].copy()
            targets = (
                start_feet * (1 - smooth[frame_index]) + end_feet * smooth[frame_index]
            )
            targets[0] = start_feet[0]
            targets[1, 2] += 0.025 * np.sin(np.pi * fractions[frame_index]) ** 2

            def residual(values):
                candidate = desired.copy()
                candidate[indices] = values
                position_error = (feet(candidate) - targets).ravel() * 40
                regularization = (values - desired[indices]) * 0.15
                return np.r_[position_error, regularization]

            solution = least_squares(
                residual,
                np.clip(desired[indices], lower + 1e-7, upper - 1e-7),
                bounds=(lower, upper),
                max_nfev=40,
            )
            bridge[frame_index, indices] = solution.x
            errors.append(
                float(np.linalg.norm(feet(bridge[frame_index])[0] - start_feet[0]))
            )
        transition_start = (len(result) - 1) * 0.02
        result = np.concatenate([result, bridge[1:], states[1:]])
        transitions.append(
            dict(
                start=transition_start,
                end=transition_start + config.transition_seconds,
                planted_foot="left",
                maximum_reference_plant_error_m=max(errors),
                source_transform_translation=shift.tolist(),
                source_transform_yaw=previous_yaw,
            )
        )
        segments.append(
            dict(name=name, start=transitions[-1]["end"], end=(len(result) - 1) * 0.02)
        )
    # Hold after the final return motion to expose delayed instability.
    result = np.concatenate([result, np.repeat(result[-1:], 75, axis=0)])
    destination = Path(config.output) / "attempts" / config.attempt
    if (destination / "actual.npz").exists():
        raise FileExistsError(f"Preserving previous episode: {destination}")
    destination.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(destination / "reference.npz", qpos=result, fps=50.0)
    (destination / "composition.json").write_text(
        json.dumps(
            dict(
                segments=segments,
                transitions=transitions,
                endpoint_ease_seconds=0.3,
                final_hold_seconds=1.5,
                selection=names,
                kick_recovery=bool(config.get("kick_recovery", False)),
                source_ranges={
                    name: list(interval)
                    for name, interval in config.get("source_ranges", {}).items()
                },
            ),
            indent=2,
        )
    )
    OmegaConf.save(config, destination / "config.yaml")
    simulate(
        config, result, destination, None if config.bag is None else list(config.bag)
    )


def render_keyframes(folder):
    """Offline evidence only: replay physical states, never reference states."""
    from PIL import Image, ImageDraw, ImageFont

    folder = Path(folder)
    archive = np.load(folder / "actual.npz")
    composition = json.loads((folder / "composition.json").read_text())
    result = json.loads((folder / "result.json").read_text())
    model = mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=720, width=1280)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = [2.08, 0.1, 0.85]
    camera.distance = 2.5
    camera.azimuth = 110
    camera.elevation = -10
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 26)
    except OSError:
        font = ImageFont.load_default()
    selections = {"transitions": [], "contacts": []}
    for transition_index, transition in enumerate(composition["transitions"]):
        times = [
            transition["start"] - 0.12,
            (transition["start"] + transition["end"]) / 2,
            transition["end"] + 0.12,
        ]
        selections["transitions"].extend(
            [
                (f"Transition {transition_index+1} | {label}", timestamp)
                for label, timestamp in zip(["before", "middle", "after"], times)
            ]
        )
    for label, body in [
        ("Foot impact", "right_ankle_roll_link"),
        ("Hand impact", "right_wrist_yaw_link"),
    ]:
        contacts = [row for row in result["contacts"] if row["body"] == body]
        peak = max(contacts, key=lambda row: row["force"])
        selections["contacts"].extend(
            [
                (f"{label} | {suffix}", peak["time"] + offset)
                for suffix, offset in [("before", -0.16), ("peak", 0), ("after", 0.25)]
            ]
        )
    for name, samples in selections.items():
        montage = Image.new("RGB", (1920, 720))
        metadata = []
        for sample_index, (label, timestamp) in enumerate(samples):
            frame_index = min(len(archive["qpos"]) - 1, max(0, round(timestamp * 50)))
            data.qpos[:] = archive["qpos"][frame_index]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera)
            tile = Image.fromarray(renderer.render())
            ImageDraw.Draw(tile).text(
                (20, 20),
                f"{label} | t={frame_index*.02:.2f}s",
                font=font,
                fill="white",
                stroke_width=1,
                stroke_fill="black",
            )
            montage.paste(
                tile.resize((640, 360)),
                ((sample_index % 3) * 640, (sample_index // 3) * 360),
            )
            metadata.append(
                dict(
                    label=label,
                    frame=frame_index,
                    time_s=frame_index * 0.02,
                    data="actual.npz:qpos",
                )
            )
        montage.save(folder / f"{name}.jpg", quality=95)
        (folder / f"{name}_frames.json").write_text(json.dumps(metadata, indent=2))
    renderer.close()
