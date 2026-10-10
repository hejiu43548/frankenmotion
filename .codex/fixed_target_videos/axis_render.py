import os

os.environ["PYOPENGL_PLATFORM"] = "egl"
os.environ["OMP_NUM_THREADS"] = "1"
import json
from pathlib import Path
import sys
import numpy as np
import torch
import smplx
import pyrender
import trimesh
import imageio.v2 as imageio
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, "/home/pku/frankenmotion/work/fixed_target_20261010/repository")
from src.tools.smplrifke_feats import smplrifkefeats_to_smpldata

root = Path("/home/pku/frankenmotion/work/fixed_target_20261010/position_sweep")
rows = json.loads((root / "commands.json").read_text())["rows"]
torch.set_num_threads(4)
model = smplx.SMPLH(
    "/home/pku/frankenmotion/deps/smplh/SMPLH_MALE.npz",
    ext="npz",
    use_pca=False,
    flat_hand_mean=True,
    num_betas=10,
).eval()
renderer = pyrender.OffscreenRenderer(400, 500)
font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
small_font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 16)
sequences = {}
for row in rows:
    archive = np.load(root / (row["key"] + ".npz"))
    motion = torch.from_numpy(archive["motion"]).float()
    decoded = smplrifkefeats_to_smpldata(motion)
    frames = len(motion)
    with torch.no_grad():
        body = model(
            global_orient=decoded["poses"][:, :3],
            body_pose=decoded["poses"][:, 3:],
            betas=torch.zeros(frames, 10),
            left_hand_pose=torch.zeros(frames, 45),
            right_hand_pose=torch.zeros(frames, 45),
        )
    vertices = (
        body.vertices.numpy()
        - body.joints.numpy()[:, :1]
        + decoded["trans"].numpy()[:, None]
    )
    joints = (
        body.joints.numpy()[:, :22]
        - body.joints.numpy()[:, :1]
        + decoded["trans"].numpy()[:, None]
    )
    side = joints[0, 1, :2] - joints[0, 2, :2]
    heading = np.arctan2(side[1], side[0]) - np.pi / 2
    rotation = np.array(
        [
            [np.cos(heading), np.sin(heading), 0],
            [-np.sin(heading), np.cos(heading), 0],
            [0, 0, 1],
        ]
    )
    vertices = vertices @ rotation.T
    joints = joints @ rotation.T
    assert abs(joints - archive["joints"][:, :22]).max() < 1e-3
    sequences[row["key"]] = (vertices, joints)
metadata = []
for axis, direction in [
    ("forward", [0.1, -1, 0.25]),
    ("lateral", [1, 0, 0.2]),
    ("height", [1, -1.5, 0.35]),
]:
    selected = [row for row in rows if row["axis"] == axis]
    scene = pyrender.Scene(
        bg_color=[0.94, 0.96, 0.98, 1], ambient_light=[0.55, 0.55, 0.55]
    )
    floor = trimesh.creation.box(extents=[20, 20, 0.01])
    floor.apply_translation([0, 0, -0.015])
    scene.add(
        pyrender.Mesh.from_trimesh(
            floor,
            material=pyrender.MetallicRoughnessMaterial(
                baseColorFactor=[0.66, 0.70, 0.74, 1], roughnessFactor=1
            ),
        )
    )
    back = np.array(direction, dtype=float)
    back /= np.linalg.norm(back)
    right = np.cross([0, 0, 1], back)
    right /= np.linalg.norm(right)
    up = np.cross(back, right)
    camera = np.eye(4)
    camera[:3, :3] = np.stack([right, up, back], 1)
    camera[:3, 3] = np.array([0.2, -0.15, 0.95]) + 3.05 * back
    scene.add(pyrender.PerspectiveCamera(yfov=np.pi / 4), pose=camera)
    scene.add(pyrender.DirectionalLight(color=np.ones(3), intensity=2.5), pose=camera)
    nodes = []
    sampled = []
    with imageio.get_writer(
        str(root / (axis + ".mp4")),
        fps=20,
        codec="libx264",
        quality=9,
        macro_block_size=1,
    ) as writer:
        for frame_index in range(frames):
            canvas = Image.new("RGB", (2000, 610), "#edf3f7")
            draw = ImageDraw.Draw(canvas)
            for column, row in enumerate(selected):
                for node in nodes:
                    scene.remove_node(node)
                nodes = []
                vertices, joints = sequences[row["key"]]
                body_mesh = trimesh.Trimesh(
                    vertices=vertices[frame_index], faces=model.faces, process=False
                )
                nodes.append(
                    scene.add(
                        pyrender.Mesh.from_trimesh(
                            body_mesh,
                            material=pyrender.MetallicRoughnessMaterial(
                                baseColorFactor=[0.25, 0.53, 0.76, 1],
                                roughnessFactor=0.8,
                            ),
                            smooth=True,
                        )
                    )
                )
                for point, radius, color in [
                    (row["target"], 0.027, [0.95, 0.1, 0.08, 1]),
                    (joints[frame_index, 21], 0.016, [0.08, 0.85, 0.25, 1]),
                ]:
                    marker = trimesh.creation.icosphere(subdivisions=2, radius=radius)
                    marker.apply_translation(point)
                    nodes.append(
                        scene.add(
                            pyrender.Mesh.from_trimesh(
                                marker,
                                material=pyrender.MetallicRoughnessMaterial(
                                    baseColorFactor=color,
                                    emissiveFactor=np.array(color[:3]) * 0.2,
                                ),
                            )
                        )
                    )
                pixels, _ = renderer.render(
                    scene, flags=pyrender.RenderFlags.SHADOWS_DIRECTIONAL
                )
                canvas.paste(Image.fromarray(pixels), (column * 400, 110))
                # Screen-space markers remain visible when the mesh occludes a point.
                for point, color, is_target in [
                    (row["target"], "#ed2418", True),
                    (joints[frame_index, 21], "#00a83d", False),
                ]:
                    local_point = camera[:3, :3].T @ (np.asarray(point) - camera[:3, 3])
                    focal = 250 / np.tan(np.pi / 8)
                    pixel_x = (
                        column * 400 + 200 + focal * local_point[0] / -local_point[2]
                    )
                    pixel_y = 110 + 250 - focal * local_point[1] / -local_point[2]
                    if is_target:
                        draw.line(
                            (pixel_x - 8, pixel_y, pixel_x + 8, pixel_y),
                            fill=color,
                            width=3,
                        )
                        draw.line(
                            (pixel_x, pixel_y - 8, pixel_x, pixel_y + 8),
                            fill=color,
                            width=3,
                        )
                    else:
                        draw.ellipse(
                            (pixel_x - 5, pixel_y - 5, pixel_x + 5, pixel_y + 5),
                            outline=color,
                            width=2,
                        )
                draw.text(
                    (column * 400 + 12, 8),
                    f"{axis.upper()} {column+1}/5 | seed 51000",
                    font=font,
                    fill="#17334b",
                )
                draw.text(
                    (column * 400 + 12, 34),
                    "XYZ = ("
                    + ", ".join(f"{value:.2f}" for value in row["target"])
                    + ") m",
                    font=font,
                    fill="#17334b",
                )
                draw.text(
                    (column * 400 + 12, 60),
                    f'Final mean error: {row["error_cm"]:.1f} cm',
                    font=font,
                    fill="#bf321d" if row["error_cm"] >= 10 else "#17334b",
                )
                draw.text(
                    (column * 400 + 12, 85),
                    f"Red: fixed target | Green: wrist | {frame_index/20:.2f}s",
                    font=small_font,
                    fill="#17334b",
                )
            writer.append_data(np.asarray(canvas))
            if frame_index in np.linspace(0, frames - 1, 6, dtype=int):
                sampled.append(canvas.resize((1600, 488)))
        canvas.save(root / (axis + "_endpoints.png"))
        draw = ImageDraw.Draw(canvas)
        draw.rectangle((0, 580, 2000, 610), fill="#edf3f7")
        draw.text(
            (15, 586),
            "FINAL FRAME FREEZE (1 second) | Same text, timing and noise; only fixed world target changes. Z = height above ground.",
            font=small_font,
            fill="#17334b",
        )
        for repeat_index in range(20):
            writer.append_data(np.asarray(canvas))
    sheet = Image.new("RGB", (1600, 488 * len(sampled)), "white")
    for sample_index, frame in enumerate(sampled):
        sheet.paste(frame, (0, sample_index * 488))
    sheet.save(root / (axis + "_inspection.jpg"))
    reader = imageio.get_reader(str(root / (axis + ".mp4")))
    meta = reader.get_meta_data()
    metadata.append(
        dict(
            axis=axis,
            frames=reader.count_frames(),
            fps=meta["fps"],
            size=meta["size"],
            source_frames=frames,
            final_freeze_frames=20,
        )
    )
    reader.close()
    print(metadata[-1], flush=True)
renderer.delete()
(root / "render_manifest.json").write_text(json.dumps(metadata, indent=2) + "\n")
