"""Fixed-camera skeleton comparison with body-relative targets in world space."""

from pathlib import Path
import subprocess

import hydra
import imageio_ffmpeg
import matplotlib

matplotlib.use("Agg")
from matplotlib import pyplot as plt
import numpy as np


@hydra.main(version_base="1.3", config_path="../config", config_name="render_strike3d")
def main(config):
    sources = [
        np.load(Path(run.directory) / f"{config.request}_{run.mode}.npz")
        for run in config.runs
    ]
    joints = [archive["joints"][config.seed_index, :, :22] for archive in sources]
    targets = [archive["target"][config.seed_index] for archive in sources]
    event_frames = [
        int(archive["event_frames"][config.seed_index]) for archive in sources
    ]
    frames = len(joints[0])
    assert all(len(positions) == frames for positions in joints)
    parents = np.load(config.skeleton)["parents"][:22]
    figure = plt.figure(figsize=(4.8 * len(joints), 4.8), dpi=100)
    axes = [
        figure.add_subplot(1, len(joints), index + 1, projection="3d")
        for index in range(len(joints))
    ]
    width, height = figure.canvas.get_width_height()
    destination = Path(config.output)
    destination.parent.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [
            imageio_ffmpeg.get_ffmpeg_exe(),
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-vcodec",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            f"{width}x{height}",
            "-r",
            "20",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
            "-vf",
            "pad=ceil(iw/2)*2:ceil(ih/2)*2",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "20",
            str(destination),
        ],
        stdin=subprocess.PIPE,
    )
    combined = np.concatenate(joints, axis=0).reshape(-1, 3)
    center = (combined[:, :2].min(0) + combined[:, :2].max(0)) / 2
    radius = max(0.8, float(np.ptp(combined[:, :2], axis=0).max()) / 2 + 0.15)
    try:
        for frame in range(frames):
            for index, axis in enumerate(axes):
                axis.clear()
                positions = joints[index][frame]
                left = positions[1] - positions[2]
                left[2] = 0
                left /= np.linalg.norm(left)
                up = np.array([0.0, 0.0, 1.0])
                forward = np.cross(left, up)
                target_world = (
                    positions[0]
                    + np.stack([forward, left, up], axis=1) @ targets[index]
                )
                for joint_index in range(1, 22):
                    segment = positions[[int(parents[joint_index]), joint_index]]
                    color = "#ce6237" if joint_index in [17, 19, 21] else "#3977ac"
                    axis.plot(
                        segment[:, 0],
                        segment[:, 1],
                        segment[:, 2],
                        color=color,
                        linewidth=2.5,
                    )
                axis.scatter(*target_world, color="red", marker="x", s=90)
                axis.scatter(*positions[21], color="#ce6237", s=35)
                axis.plot(
                    joints[index][: frame + 1, 21, 0],
                    joints[index][: frame + 1, 21, 1],
                    joints[index][: frame + 1, 21, 2],
                    color="#ce6237",
                    alpha=0.3,
                )
                axis.set_xlim(center[0] - radius, center[0] + radius)
                axis.set_ylim(center[1] - radius, center[1] + radius)
                axis.set_zlim(0, max(1.8, float(combined[:, 2].max()) + 0.1))
                axis.set_box_aspect((1, 1, 1.2))
                axis.view_init(elev=15, azim=-65)
                axis.set_xlabel("World X (m)")
                axis.set_ylabel("World Y (m)")
                axis.set_title(
                    config.runs[index].label
                    + (
                        "\nEndpoint frame"
                        if frame == event_frames[index]
                        else f"\nt = {frame/20:.2f}s"
                    )
                )
            figure.suptitle(
                f'{config.request}, seed {int(sources[0]["seeds"][config.seed_index])} | red cross: requested body-relative wrist target'
            )
            figure.tight_layout()
            figure.canvas.draw()
            image = np.asarray(figure.canvas.buffer_rgba())[..., :3].copy()
            process.stdin.write(image.tobytes())
            if frame == event_frames[0]:
                figure.savefig(destination.with_suffix(".png"))
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError("ffmpeg render failed")
    finally:
        plt.close(figure)
    print(destination)


if __name__ == "__main__":
    main()
