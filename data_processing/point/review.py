"""Render real XYZ reach crops and their fixed spatial targets."""

import html
import json
from pathlib import Path
import subprocess

import hydra
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import torch

from shared_motion.training.geometry import Skeleton
from shared_motion.training.model import file_sha256
from shared_motion.training.runner import save_json


def draw_frame(joints, parents, frame, entry, number, reason):
    image = Image.new("RGB", (1100, 680), "#f4f7fa")
    drawing = ImageDraw.Draw(image)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    record = entry["record"]
    drawing.text(
        (16, 12),
        f'#{number} REACH XYZ | target {tuple(round(value, 3) for value in record["target_xyz_m"])} m',
        font=font,
        fill="#183248",
    )
    drawing.text((16, 40), record["family"], font=small, fill="#183248")
    drawing.text(
        (16, 65),
        f'Source {record["crop_start_frame_20fps"]/20:.2f}-{record["crop_end_frame_20fps"]/20:.2f}s | frame {frame+1}/{len(joints)} | unchanged crop',
        font=small,
        fill="#526477",
    )
    projections = [
        np.array([[1, 0, 0], [0, 0, -1]]),
        np.array([[0.707, -0.707, 0], [0.354, 0.354, -0.866]]),
    ]
    lateral = joints[0, 1, :2] - joints[0, 2, :2]
    lateral /= np.linalg.norm(lateral)
    forward = np.array([lateral[1], -lateral[0]])
    target = joints[0, 0].copy()
    target[:2] += (
        record["target_xyz_m"][0] * forward + record["target_xyz_m"][1] * lateral
    )
    target[2] += record["target_xyz_m"][2]
    for column, projection in enumerate(projections):
        projected = joints @ projection.T
        center = (projected.max((0, 1)) + projected.min((0, 1))) / 2
        extent = np.ptp(projected, axis=(0, 1))
        scale = min(240, 490 / max(extent[0], 0.1), 440 / max(extent[1], 0.1))
        origin = np.array([column * 550 + 275, 355]) - center * scale
        positions = projected[frame] * scale + origin
        drawing.text(
            (column * 550 + 16, 100),
            ["SIDE | forward +X ->", "OBLIQUE | fixed camera"][column],
            font=small,
            fill="#526477",
        )
        for joint_index in range(1, 24):
            color = (
                "#d94b43"
                if joint_index in {1, 4, 7, 10, 13, 16, 18, 20, 22}
                else "#2474ba"
            )
            drawing.line(
                [tuple(positions[parents[joint_index]]), tuple(positions[joint_index])],
                fill=color,
                width=4,
            )
        target_pixel = target @ projection.T * scale + origin
        drawing.ellipse(
            [tuple(target_pixel - 7), tuple(target_pixel + 7)],
            outline="#b52dac",
            width=3,
        )
        wrist_trace = projected[: frame + 1, 21] * scale + origin
        if len(wrist_trace) > 1:
            drawing.line(
                [tuple(point) for point in wrist_trace], fill="#b52dac", width=2
            )
        trace = projected[:, 0] * scale + origin
        drawing.line([tuple(point) for point in trace], fill="#adbaca", width=1)
    metrics = record["metrics"]
    drawing.text(
        (16, 610),
        f'Wrist approach {metrics["right_wrist_displacement_m"]:.3f} m | root drift {metrics["root_excursion_m"]:.3f} m | hold speed {metrics["hold_speed_m_s"]:.3f} m/s',
        font=small,
        fill="#526477",
    )
    drawing.text(
        (16, 640),
        "Purple circle = final XYZ target; purple trace = wrist path | 20fps | not trained",
        font=small,
        fill="#526477",
    )
    return image


@hydra.main(
    version_base="1.3", config_path="../../config", config_name="reach_xyz_repair"
)
def main(config):
    torch.set_num_threads(2)
    output = Path(config.output)
    finalized = (output / "reach_index.json").exists()
    records = json.loads(
        (output / ("reach_index.json" if finalized else "inspection.json")).read_text()
    )
    review = output / ("review" if finalized else "preview")
    review.mkdir(exist_ok=True)
    skeleton = Skeleton(config.skeleton)
    candidates = [record for record in records if record["split"] == "train"]
    selected = []
    families = set()
    # Three-dimensional farthest-point sampling, after prioritizing HDM evidence.
    for record in candidates:
        if (
            record["family"].startswith("MPI_HDM05")
            and record["family"] not in families
        ):
            selected.append(record)
            families.add(record["family"])
            if len(selected) == 2:
                break
    while len(selected) < min(10, len({record["family"] for record in candidates})):
        pool = [record for record in candidates if record["family"] not in families]
        record = max(
            pool,
            key=lambda record: (
                min(
                    np.linalg.norm(
                        np.array(record["target_xyz_m"])
                        - np.array(other["target_xyz_m"])
                    )
                    for other in selected
                )
                if selected
                else 0
            ),
        )
        selected.append(record)
        families.add(record["family"])
    items = []
    for number, record in enumerate(selected, 1):
        motion = np.load(record["motion_source"], mmap_mode="r")[
            record["crop_start_frame_20fps"] : record["crop_end_frame_20fps"]
        ].copy()
        with torch.no_grad():
            joints = skeleton(torch.from_numpy(motion)[None])[0].numpy()
        video = review / f"{number:02d}_reach_xyz.mp4"
        process = subprocess.Popen(
            [
                "ffmpeg",
                "-y",
                "-loglevel",
                "error",
                "-f",
                "rawvideo",
                "-pix_fmt",
                "rgb24",
                "-s",
                "1100x680",
                "-r",
                "20",
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(video),
            ],
            stdin=subprocess.PIPE,
        )
        for frame in range(len(joints)):
            process.stdin.write(
                draw_frame(
                    joints,
                    skeleton.parents + [20, 21],
                    frame,
                    {"record": record},
                    number,
                    "",
                ).tobytes()
            )
        process.stdin.close()
        if process.wait():
            raise RuntimeError("ffmpeg failed")
        keyframes = np.linspace(0, len(joints) - 1, 6, dtype=int).tolist()
        contact = Image.new("RGB", (1650, 680), "white")
        for position, frame in enumerate(keyframes):
            contact.paste(
                draw_frame(
                    joints,
                    skeleton.parents + [20, 21],
                    frame,
                    {"record": record},
                    number,
                    "",
                ).resize((550, 340)),
                ((position % 3) * 550, (position // 3) * 340),
            )
        contact.save(review / f"{number:02d}_contact.png")
        probe = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=nb_frames,r_frame_rate",
                    "-of",
                    "json",
                    str(video),
                ]
            )
        )["streams"][0]
        assert (
            int(probe["nb_frames"]) == len(joints) and probe["r_frame_rate"] == "20/1"
        )
        items.append(
            dict(
                record=record,
                video=video.name,
                sha256=file_sha256(video),
                keyframes=keyframes,
            )
        )
    cards = "".join(
        f'<article><h2>#{number+1} XYZ {tuple(round(value,3) for value in item["record"]["target_xyz_m"])} m</h2><p>{html.escape(item["record"]["family"])}</p><p>{html.escape(item["record"]["original_caption"])}</p><video controls loop preload="metadata" src="{item["video"]}"></video></article>'
        for number, item in enumerate(items)
    )
    (review / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Reach XYZ训练数据复核</title><style>body{font:16px system-ui;max-width:1100px;margin:28px auto;background:#f3f6f9}article{padding:20px;background:white;margin:20px 0}video{width:100%}</style><h1>point合并入reach：右手腕XYZ目标</h1><p>初始骨盆为原点，X向前、Y向左、Z向上，单位米。紫圈为末尾停留目标，紫线为腕部轨迹。真实裁剪，无改速或轨迹变形。未训练，待复核。</p><p><a href="manifest.json">来源、XYZ和准入指标</a></p>'
        + cards
    )
    save_json(
        review / "manifest.json",
        dict(
            items=items,
            training_started=False,
            finalized=finalized,
            selection="up to2 HDM families, then farthest-point XYZ target diversity; distinct TRAIN families",
        ),
    )
    print(json.dumps(dict(review=str(review), videos=len(items))), flush=True)


if __name__ == "__main__":
    main()
