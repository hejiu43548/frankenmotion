"""Saved-state diagnostic figures; never used as controller input."""

import json
import os
from pathlib import Path
import sys

os.environ.setdefault("MUJOCO_GL", "egl")
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.transform import Rotation


def main(folder):
    folder = Path(folder)
    archive = np.load(folder / "actual.npz")
    states = archive["qpos"]
    reference = archive["reference"]
    composition = json.loads((folder / "composition.json").read_text())
    semantics = np.load(folder / "semantic_timeseries.npz")
    time = semantics["time_s"]
    rotations = Rotation.from_quat(states[:, [4, 5, 6, 3]])
    yaw = np.rad2deg(np.unwrap(rotations.as_euler("xyz")[:, 2]))
    names = archive["body_names"].tolist()
    wrist = rotations.inv().apply(
        archive["body_pos"][:, names.index("right_wrist_yaw_link")]
        - archive["body_pos"][:, names.index("pelvis")]
    )
    figure, axes = plt.subplots(2, 2, figsize=(12, 9))
    axis = axes[0, 0]
    axis.plot(
        reference[:, 0], reference[:, 1], "--", color="gray", label="Composed reference"
    )
    axis.plot(states[:, 0], states[:, 1], color="black", label="Actual physics")
    for segment in composition["segments"]:
        first, last = [round(segment[name] * 50) for name in ["start", "end"]]
        axis.plot(
            states[first : last + 1, 0],
            states[first : last + 1, 1],
            label=segment["name"],
        )
    axis.add_patch(
        plt.Circle(states[0, :2], 0.3, fill=False, color="green", label="0.30 m goal")
    )
    axis.scatter(*states[0, :2], marker="o", color="green")
    axis.scatter(*states[-1, :2], marker="x", color="red")
    axis.axis("equal")
    axis.set(
        xlabel="world X (m)",
        ylabel="world Y (m)",
        title="Actual return 0.204 m; reference is not closed",
    )
    axis.legend(fontsize=8)
    axes[0, 1].plot(time, yaw, label="Actual root yaw")
    axes[0, 1].axhline(180, color="green", linestyle="--", label="180 degree target")
    axes[0, 1].set(
        xlabel="time (s)",
        ylabel="degrees",
        title="Moving turn, then curved return walk",
    )
    axes[0, 1].legend()
    axes[1, 0].plot(time, semantics["torso_pitch_deg"], label="Torso forward pitch")
    axes[1, 0].set(xlabel="time (s)", ylabel="degrees", title="Bow and recovery")
    axes[1, 1].plot(time, wrist[:, 1], label="Right wrist lateral")
    axes[1, 1].plot(time, wrist[:, 2], label="Right wrist height")
    axes[1, 1].set(
        xlabel="time (s)", ylabel="pelvis-relative m", title="Repeated waving"
    )
    axes[1, 1].legend()
    for axis in [axes[0, 1], axes[1, 0], axes[1, 1]]:
        for transition in composition["transitions"]:
            axis.axvspan(
                transition["start"], transition["end"], alpha=0.12, color="gray"
            )
    figure.tight_layout()
    figure.savefig(folder / "trajectory_semantics.png", dpi=150)
    plt.close(figure)
    model = mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=720, width=1280)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 2.7
    camera.azimuth = 125
    camera.elevation = -12
    font = ImageFont.truetype("DejaVuSans.ttf", 24)
    wave = next(
        segment for segment in composition["segments"] if segment["name"] == "wave"
    )
    bow = next(
        segment for segment in composition["segments"] if segment["name"] == "bow"
    )
    first, last = [round(bow[name] * 50) for name in ["start", "end"]]
    peak = first + int(np.argmax(semantics["torso_pitch_deg"][first : last + 1]))
    groups = {
        "wave_detail": [
            round(timestamp * 50)
            for timestamp in np.linspace(wave["start"] + 1, wave["end"] - 0.5, 9)
        ],
        "bow_detail": [first, peak, last],
    }
    for name, indices in groups.items():
        montage = Image.new("RGB", (1920, 360 * ((len(indices) + 2) // 3)))
        metadata = []
        for index, frame in enumerate(indices):
            data.qpos[:] = states[frame]
            mujoco.mj_forward(model, data)
            camera.lookat[:] = [*states[frame, :2], 0.85]
            renderer.update_scene(data, camera)
            tile = Image.fromarray(renderer.render())
            ImageDraw.Draw(tile).text(
                (20, 20),
                f"ACTUAL | {name} | {frame/50:.2f}s | torso {semantics['torso_pitch_deg'][frame]:.1f} deg",
                font=font,
                fill="white",
                stroke_width=1,
                stroke_fill="black",
            )
            montage.paste(
                tile.resize((640, 360)), ((index % 3) * 640, (index // 3) * 360)
            )
            metadata.append(
                dict(frame=frame, time_s=frame / 50, source="actual.npz:qpos")
            )
        montage.save(folder / f"{name}.jpg", quality=95)
        (folder / f"{name}_frames.json").write_text(json.dumps(metadata, indent=2))
    renderer.close()


if __name__ == "__main__":
    main(sys.argv[1])
