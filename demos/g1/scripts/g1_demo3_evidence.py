"""Plot actual closure and support; render saved-state start and end views."""

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
    metrics = json.loads((folder / "metrics.json").read_text())
    time = np.arange(len(states)) / 50
    yaw = np.rad2deg(
        np.unwrap(Rotation.from_quat(states[:, [4, 5, 6, 3]]).as_euler("xyz")[:, 2])
    )
    reference_yaw = np.rad2deg(
        np.unwrap(Rotation.from_quat(reference[:, [4, 5, 6, 3]]).as_euler("xyz")[:, 2])
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
    axis.scatter(*states[0, :2], marker="o", color="green", label="Start")
    axis.scatter(*states[-1, :2], marker="x", color="red", label="End after hold")
    axis.axis("equal")
    axis.set(
        xlabel="world X (m)",
        ylabel="world Y (m)",
        title=f"Actual final return {metrics['return_distance_m']:.3f} m",
    )
    axis.legend(fontsize=8)
    axes[0, 1].plot(time, yaw - yaw[0], label="Actual yaw change")
    axes[0, 1].plot(
        time, reference_yaw - reference_yaw[0], "--", label="Reference yaw change"
    )
    axes[0, 1].axhline(360, color="green", linestyle=":", label="360 degree target")
    axes[0, 1].set(
        xlabel="time (s)",
        ylabel="degrees",
        title=f"Two left turns; final error {metrics['heading_error_deg']:.2f} deg",
    )
    axes[0, 1].legend()
    axes[1, 0].plot(
        time,
        np.linalg.norm(states[:, :2] - states[0, :2], axis=1),
        label="Distance from actual start",
    )
    axes[1, 0].plot(
        time,
        np.linalg.norm(states[:, :2] - reference[:, :2], axis=1),
        label="Actual-reference error",
    )
    axes[1, 0].axhline(0.3, color="green", linestyle=":")
    axes[1, 0].set(xlabel="time (s)", ylabel="m", title="Closure and tracker drift")
    axes[1, 0].legend()
    axes[1, 1].plot(time[:-1], archive["telemetry"][:, 0], label="Left foot")
    axes[1, 1].plot(time[:-1], archive["telemetry"][:, 1], label="Right foot")
    axes[1, 1].set(
        xlabel="time (s)", ylabel="N", title="Actual support forces (20 ms mean)"
    )
    axes[1, 1].legend()
    for axis in [axes[0, 1], axes[1, 0], axes[1, 1]]:
        for transition in composition["transitions"]:
            axis.axvspan(
                transition["start"], transition["end"], alpha=0.12, color="gray"
            )
    figure.tight_layout()
    figure.savefig(folder / "trajectory.png", dpi=150)
    plt.close(figure)
    model = mujoco.MjModel.from_xml_path(str(folder / "scene.xml"))
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=720, width=1280)
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance = 3.5
    camera.azimuth = 125
    camera.elevation = -35
    camera.lookat[:] = [*states[0, :2], 0.65]
    font = ImageFont.truetype("DejaVuSans.ttf", 25)
    montage = Image.new("RGB", (2560, 720))
    for index, frame in enumerate([0, len(states) - 1]):
        data.qpos[:] = states[frame]
        mujoco.mj_forward(model, data)
        renderer.update_scene(data, camera)
        tile = Image.fromarray(renderer.render())
        ImageDraw.Draw(tile).text(
            (20, 20),
            f"ACTUAL | {'START' if index == 0 else 'END AFTER 2 s HOLD'} | {frame/50:.2f}s",
            font=font,
            fill="white",
            stroke_width=1,
            stroke_fill="black",
        )
        montage.paste(tile, (1280 * index, 0))
    montage.save(folder / "start_end.jpg", quality=95)
    renderer.close()


if __name__ == "__main__":
    main(sys.argv[1])
