"""Render unchanged jump crops, verify caches and show ten distinct TRAIN sources."""

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

from data_processing.jump.rules import check_clip
from shared_motion.training.catalog import HUMAN_HEIGHT
from shared_motion.training.catalog import measure
from shared_motion.training.geometry import Skeleton
from shared_motion.training.model import file_sha256
from shared_motion.training.runner import save_json


def draw_frame(joints, parents, frame, record, number):
    image = Image.new("RGB", (1000, 650), "#f4f7fa")
    drawing = ImageDraw.Draw(image)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 19)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    metrics = record["metrics"]
    drawing.text(
        (18, 12),
        f'#{number:02d} IN-PLACE TWO-FOOT JUMP | TRAIN | rise {metrics["root_rise_m"]:.3f} m',
        font=font,
        fill="#183248",
    )
    drawing.text((18, 42), record["family"], font=small, fill="#183248")
    drawing.text(
        (18, 65),
        f'Source {record["crop_start_frame_20fps"]/20:.2f}-{record["crop_end_frame_20fps"]/20:.2f}s | {record["annotation_level"]} | frame {frame+1}/{len(joints)}',
        font=small,
        fill="#526477",
    )
    left_joints = {1, 4, 7, 10, 13, 16, 18, 20, 22}
    right_joints = {2, 5, 8, 11, 14, 17, 19, 21, 23}
    floor = float(np.median(np.minimum(joints[:3, [7, 8], 2], joints[:3, [10, 11], 2])))
    projections = [
        np.array([[0, 1, 0], [0, 0, -1]]),
        np.array([[0.707, -0.707, 0], [0.354, 0.354, -0.866]]),
    ]
    for column, projection in enumerate(projections):
        projected = joints @ projection.T
        midpoint = (projected.min((0, 1)) + projected.max((0, 1))) / 2
        extent = np.ptp(projected, axis=(0, 1))
        scale = min(245, 425 / max(extent[0], 0.1), 410 / max(extent[1], 0.1))
        origin = np.array([column * 500 + 250, 345]) - midpoint * scale
        positions = projected[frame] * scale + origin
        drawing.text(
            (column * 500 + 18, 102),
            ["FRONT | red=left / blue=right", "OBLIQUE | fixed camera"][column],
            font=small,
            fill="#526477",
        )
        for offset in np.arange(-1.0, 1.01, 0.25):
            for axis in [0, 1]:
                endpoints = np.array([[-1.0, offset, floor], [1.0, offset, floor]])
                if axis:
                    endpoints[:, [0, 1]] = endpoints[:, [1, 0]]
                points = endpoints @ projection.T * scale + origin
                drawing.line(
                    [tuple(point) for point in points], fill="#d8e1e9", width=1
                )
        for joint_index in range(1, 24):
            color = (
                "#d94b43"
                if joint_index in left_joints
                else "#2474ba" if joint_index in right_joints else "#354554"
            )
            drawing.line(
                [tuple(positions[parents[joint_index]]), tuple(positions[joint_index])],
                fill=color,
                width=4,
            )
        for joint_index in [7, 8, 10, 11]:
            point = positions[joint_index]
            drawing.ellipse(
                (point[0] - 4, point[1] - 4, point[0] + 4, point[1] + 4), fill="#23384a"
            )
    phase = (
        "BOTH FEET AIRBORNE"
        if metrics["flight_start"] <= frame < metrics["flight_stop"]
        else "PREPARATION / LANDING"
    )
    drawing.text(
        (18, 588),
        f'{phase} | takeoff / landing gap: {metrics["takeoff_difference_frames"]} / {metrics["landing_difference_frames"]} frames',
        font=small,
        fill="#183248",
    )
    drawing.text(
        (18, 617),
        f'Landing offset: center {metrics["landing_center_offset_m"]*100:.1f} cm / each foot {metrics["each_foot_landing_offset_m"]*100:.1f} cm | raw crop, 20 fps',
        font=small,
        fill="#526477",
    )
    return image


def select_examples(records):
    training = [record for record in records if record["split"] == "train"]
    selected = []
    families = set()
    # Include available HDM05 first; then cover the measured height distribution.
    for record in training:
        if (
            record["family"].startswith("MPI_HDM05/")
            and record["family"] not in families
        ):
            selected.append(record)
            families.add(record["family"])
    targets = np.linspace(
        min(record["metrics"]["root_rise_m"] for record in training),
        max(record["metrics"]["root_rise_m"] for record in training),
        10 - len(selected),
    )
    for target in targets:
        choices = [record for record in training if record["family"] not in families]
        record = min(
            choices,
            key=lambda entry: (
                abs(entry["metrics"]["root_rise_m"] - target),
                entry["family"],
                entry["key"],
            ),
        )
        selected.append(record)
        families.add(record["family"])
    assert len(selected) == len(families) == 10
    return selected


@hydra.main(version_base="1.3", config_path="../../config", config_name="jump_repair")
def main(config):
    torch.set_num_threads(2)
    output = Path(config.output)
    finalized = (output / "jump_index.json").exists()
    records = json.loads(
        (output / ("jump_index.json" if finalized else "inspection.json")).read_text()
    )
    review = output / ("review" if finalized else "preview")
    review.mkdir(exist_ok=True)
    skeleton = Skeleton(config.skeleton)
    maximum_error = 0.0
    for record in records:
        begin, end = record["crop_start_frame_20fps"], record["crop_end_frame_20fps"]
        source_crop = np.load(record["motion_source"], mmap_mode="r")[begin:end].copy()
        with torch.no_grad():
            joints = skeleton(torch.from_numpy(source_crop)[None])[0].numpy() * (
                HUMAN_HEIGHT / skeleton.height
            )
        assert not check_clip(joints, config.rules)[1], record["key"]
        if finalized:
            with np.load(record["cache"]) as archive:
                assert np.array_equal(source_crop, archive["motion"])
                assert (
                    len(source_crop) == record["real_frames"]
                    and record["pad_frames"] == 0
                )
                assert file_sha256(record["cache"]) == record["cache_sha256"]
                with torch.no_grad():
                    value = float(
                        measure(
                            skeleton,
                            torch.from_numpy(source_crop)[None],
                            torch.tensor([8]),
                            torch.tensor([len(source_crop)]),
                        )[0]
                    )
                maximum_error = max(
                    maximum_error, abs(value - float(archive["quantity"]))
                )
        assert all(np.isfinite(source_crop).flat)
    assert maximum_error < 1e-6
    if finalized:
        manifests = {}
        for split in ["train", "val"]:
            before = json.loads((Path(config.base) / f"{split}.json").read_text())
            after = json.loads((output / f"{split}.json").read_text())
            assert [record for record in before if record["task"] != "jump"] == [
                record for record in after if record["task"] != "jump"
            ]
            manifests[split] = after
        assert not (
            {record["family"] for record in manifests["train"]}
            & {record["family"] for record in manifests["val"]}
        )
    selected = select_examples(records)
    items = []
    for number, record in enumerate(selected, 1):
        clip = np.load(record["motion_source"], mmap_mode="r")[
            record["crop_start_frame_20fps"] : record["crop_end_frame_20fps"]
        ].copy()
        with torch.no_grad():
            joints = skeleton(torch.from_numpy(clip)[None])[0].numpy() * (
                HUMAN_HEIGHT / skeleton.height
            )
        video = review / f"{number:02d}_jump_v3.mp4"
        encoder = subprocess.Popen(
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
                "1000x650",
                "-r",
                "20",
                "-i",
                "-",
                "-an",
                "-c:v",
                "libx264",
                "-threads",
                "2",
                "-preset",
                "fast",
                "-crf",
                "21",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(video),
            ],
            stdin=subprocess.PIPE,
        )
        for frame in range(len(joints)):
            encoder.stdin.write(
                draw_frame(
                    joints, skeleton.parents + [20, 21], frame, record, number
                ).tobytes()
            )
        encoder.stdin.close()
        assert encoder.wait() == 0
        metrics = record["metrics"]
        keyframes = [
            0,
            max(0, metrics["flight_start"] - 1),
            int(joints[:, 0, 2].argmax()),
            min(len(joints) - 1, metrics["flight_stop"]),
            len(joints) - 1,
        ]
        sheet = Image.new("RGB", (2500, 325), "white")
        for column, frame in enumerate(keyframes):
            sheet.paste(
                draw_frame(
                    joints, skeleton.parents + [20, 21], frame, record, number
                ).resize((500, 325)),
                (column * 500, 0),
            )
        sheet.save(review / f"{number:02d}_contact.png")
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
                number=number,
                video=video.name,
                sha256=file_sha256(video),
                keyframes=keyframes,
                record=record,
            )
        )
    save_json(
        review / "manifest.json",
        dict(
            sampling="HDM05 if available, then height-stratified selection of ten distinct TRAIN recordings; not a random success-rate estimate",
            items=items,
        ),
    )
    save_json(
        review / "verification.json",
        dict(
            finalized=finalized,
            caches=len(records),
            exact_source_crops=finalized,
            quantity_max_error=maximum_error,
            other19_unchanged=finalized,
            split_isolation=finalized,
            videos=10,
            frames_and_20fps_verified=True,
            training_started=False,
        ),
    )
    sections = "".join(
        f'<article><h2>#{entry["number"]:02d} · {entry["record"]["metrics"]["root_rise_m"]:.3f} m</h2><p>{html.escape(entry["record"]["family"])}</p><p>{html.escape(entry["record"]["original_caption"])} · {entry["record"]["annotation_level"]}</p><video controls loop preload="metadata" src="{entry["video"]}"></video></article>'
        for entry in items
    )
    (review / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>jump 原地双脚跳复核 V3</title><style>body{font:16px system-ui;max-width:1050px;margin:28px auto;background:#f3f6f9}article{padding:20px;background:white;margin:20px 0}video{width:100%}</style><h1>原地双脚跳：10 条训练样本（V3）</h1><p>10 个不同原始录制，按实际跳高分层选取；来源与标注类型见各视频。要求落回起跳位置：落点中心偏差≤8cm，每脚偏差≤10cm；排除左右及前后方向跳。原始动作只裁剪，20fps，无轨迹修正或重定时。红=左，蓝=右；地面网格为支撑高度估计。未启动训练，待人工复核。</p><p><a href="manifest.json">来源、标注与裁剪</a> · <a href="verification.json">核验记录</a></p>'
        + sections
    )
    print(
        json.dumps(dict(review=str(review), count=len(records), videos=10)), flush=True
    )


if __name__ == "__main__":
    main()
