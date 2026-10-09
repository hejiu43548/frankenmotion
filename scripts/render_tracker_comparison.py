"""Render reference and recorded native-MuJoCo rollout without resimulation."""

import json
from pathlib import Path

import hydra
import imageio.v2 as imageio
import mujoco
import numpy as np
from omegaconf import DictConfig
from PIL import Image
from PIL import ImageDraw


@hydra.main(
    config_path="../config/tracker_rl", config_name="render", version_base="1.3"
)
def main(configuration: DictConfig):
    output = Path(configuration.output)
    output.mkdir(parents=True, exist_ok=False)
    model = mujoco.MjModel.from_binary_path(configuration.scene)
    model.vis.global_.offwidth = configuration.width
    model.vis.global_.offheight = configuration.height
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, configuration.height, configuration.width)
    camera = mujoco.MjvCamera()
    camera.distance = 3.8
    camera.azimuth = 135
    camera.elevation = -15
    metrics = json.loads((Path(configuration.rollouts) / "metrics.json").read_text())
    records = json.loads(
        (
            Path(configuration.artifacts)
            / "motion"
            / configuration.split
            / "clips.json"
        ).read_text()
    )["records"]
    index = []
    for record in records:
        if (
            configuration.motion_seed is not None
            and record["seed"] != configuration.motion_seed
        ):
            continue
        stem = f'{record["task"]}_{record["seed"]}'
        reference = np.load(
            Path(configuration.artifacts)
            / "motion"
            / configuration.split
            / f"{stem}.npz"
        )["qpos"]
        states = np.load(Path(configuration.rollouts) / f"{stem}.npz")["states"]
        episode = next(
            item
            for item in metrics["episodes"]
            if item["task"] == record["task"] and item["motion_seed"] == record["seed"]
        )
        video = output / f"{stem}.mp4"
        with imageio.get_writer(video, fps=25, codec="libx264", quality=7) as writer:
            for frame_index in range(0, len(reference), 2):
                panels = []
                camera.lookat[:] = reference[frame_index, :3]
                camera.lookat[2] = 0.9
                for label, sequence in [
                    ("Official motion reference", reference),
                    (configuration.label, states),
                ]:
                    ended = frame_index >= len(sequence)
                    data.qpos[:] = sequence[min(frame_index, len(sequence) - 1)]
                    mujoco.mj_forward(model, data)
                    renderer.update_scene(data, camera=camera)
                    panel = Image.fromarray(renderer.render())
                    draw = ImageDraw.Draw(panel)
                    draw.rectangle((0, 0, configuration.width, 42), fill="black")
                    draw.text((10, 6), label, fill="white")
                    draw.text(
                        (10, 23),
                        f'{record["task"]} | {frame_index / 50:.2f}s | seed {record["seed"]}',
                        fill="white",
                    )
                    if ended:
                        draw.rectangle(
                            (
                                0,
                                configuration.height - 40,
                                configuration.width,
                                configuration.height,
                            ),
                            fill="#801010",
                        )
                        draw.text(
                            (10, configuration.height - 28),
                            "TERMINATED: last recorded frame held",
                            fill="white",
                        )
                    panels.append(np.asarray(panel))
                writer.append_data(np.concatenate(panels, axis=1))
        index.append(
            {"video": video.name, "episode": episode, "prompt": record["prompt"]}
        )
        print("RENDERED", video, flush=True)
    renderer.close()
    (output / "index.json").write_text(json.dumps(index, indent=2))


if __name__ == "__main__":
    main()
