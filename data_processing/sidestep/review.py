"""Verify every cache and render10 source-left / training-right pairs."""

import html
import json
from pathlib import Path
import random
import subprocess

import hydra
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import torch

from data_processing.sidestep.rules import check_clip
from shared_motion.training.catalog import HUMAN_HEIGHT
from shared_motion.training.catalog import measure
from shared_motion.training.data import MotionDataset
from shared_motion.training.data import assert_disjoint
from shared_motion.training.geometry import Skeleton
from shared_motion.training.model import file_sha256
from shared_motion.training.runner import save_json


def draw_frame(pair, parents, frame, number, record):
    canvas = Image.new("RGB", (1200, 760), "#f5f8fb")
    drawing = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 19)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    drawing.text(
        (16, 10),
        f'#{number:02d} HDM05 walkLeft | {record["family"]}',
        font=font,
        fill="#17324a",
    )
    drawing.text(
        (16, 38),
        f'Source {record["crop_start_frame_20fps"]/20:.2f}-{record["crop_end_frame_20fps"]/20:.2f}s | t={frame/20:.2f}s | frame {frame+1}/{len(pair[0])}',
        font=small,
        fill="#536477",
    )
    labels = [
        "ORIGINAL: anatomical LEFT (reference)",
        "MIRRORED: anatomical RIGHT (TRAIN)",
    ]
    left_joints = {1, 4, 7, 10, 13, 16, 18, 20, 22}
    right_joints = {2, 5, 8, 11, 14, 17, 19, 21, 23}
    combined = np.concatenate(pair)
    horizontal_extent = max(1.8, float(np.ptp(combined[:, :, 1])))
    scale = min(230, 490 / horizontal_extent)
    vertical_midpoint = float((combined[:, :, 2].max() + combined[:, :, 2].min()) / 2)
    for column, joints in enumerate(pair):
        drawing.text((column * 600 + 16, 72), labels[column], font=font, fill="#17324a")
        distance = (
            record["native_left_directed_quantity"]
            if column == 0
            else record["quantity"]
        )
        drawing.text(
            (column * 600 + 16, 100),
            f"Directed distance: {distance:.3f} m | red left / blue right",
            font=small,
            fill="#536477",
        )
        lateral_midpoint = float((joints[:, :, 1].min() + joints[:, :, 1].max()) / 2)

        def project(points):
            return [
                tuple(point)
                for point in np.stack(
                    [
                        -(points[..., 1] - lateral_midpoint) * scale
                        + column * 600
                        + 300,
                        -(points[..., 2] - vertical_midpoint) * scale + 340,
                    ],
                    axis=-1,
                ).reshape(-1, 2)
            ]

        floor = float(joints[:, :, [2]].min())
        drawing.line(
            [(column * 600 + 20, 520), (column * 600 + 580, 520)],
            fill="#d9e3ec",
            width=1,
        )
        for joint_index in range(1, 24):
            color = (
                "#d54b42"
                if joint_index in left_joints
                else "#2375ba" if joint_index in right_joints else "#354554"
            )
            drawing.line(
                project(joints[frame, [parents[joint_index], joint_index]]),
                fill=color,
                width=5,
            )
        drawing.text(
            (column * 600 + 16, 548),
            "TOP VIEW: dark root / red left ankle / blue right ankle",
            font=small,
            fill="#536477",
        )
        for joint_index, color in [(0, "#354554"), (7, "#d54b42"), (8, "#2375ba")]:
            positions = joints[:, joint_index, :2]
            trace = np.stack(
                [
                    -(positions[:, 1] - lateral_midpoint) * scale + column * 600 + 300,
                    -positions[:, 0] * 150 + 650,
                ],
                axis=-1,
            )
            drawing.line([tuple(point) for point in trace], fill="#d6dfe7", width=1)
            if frame:
                drawing.line(
                    [tuple(point) for point in trace[: frame + 1]], fill=color, width=3
                )
            horizontal, vertical = trace[frame]
            drawing.ellipse(
                (horizontal - 4, vertical - 4, horizontal + 4, vertical + 4), fill=color
            )
        drawing.text(
            (column * 600 + 16, 712),
            "Fixed cameras / raw20fps / supplemented crop boundaries",
            font=small,
            fill="#536477",
        )
    drawing.line([(600, 65), (600, 735)], fill="#d0dce5", width=2)
    return canvas


@hydra.main(
    version_base="1.3", config_path="../../config", config_name="sidestep_repair"
)
def main(config):
    output = Path(config.output)
    review = output / "review"
    review.mkdir(exist_ok=False)
    records = json.loads((output / "sidestep_index.json").read_text())
    skeleton = Skeleton(config.skeleton)
    torch.set_num_threads(2)
    quantity_error = 0.0
    source_motion = {}
    for record in records:
        for direction, cache_field, source_field in [
            ("right", "cache", "motion_source"),
            ("left", "native_left_cache", "native_motion_source"),
        ]:
            cache_path = record[cache_field]
            assert file_sha256(cache_path) == record[cache_field + "_sha256"]
            with np.load(cache_path) as archive:
                motion = archive["motion"]
                if record[source_field] not in source_motion:
                    source_motion[record[source_field]] = np.load(record[source_field])
                assert np.array_equal(
                    motion,
                    source_motion[record[source_field]][
                        record["crop_start_frame_20fps"] : record[
                            "crop_end_frame_20fps"
                        ]
                    ],
                )
                with torch.no_grad():
                    joints = skeleton(torch.from_numpy(motion)[None])[0].numpy()
                    quantity = (
                        float(
                            measure(
                                skeleton,
                                torch.from_numpy(motion)[None],
                                torch.tensor([5]),
                                torch.tensor([len(motion)]),
                            )[0]
                        )
                        if direction == "right"
                        else float(
                            (joints[:, 0, 1] - joints[0, 0, 1]).max()
                            * HUMAN_HEIGHT
                            / skeleton.height
                        )
                    )
                metrics, failures = check_clip(joints, direction, config.rules)
                assert not failures, (record["key"], direction, failures)
                quantity_error = max(
                    quantity_error, abs(quantity - float(archive["quantity"]))
                )
    assert quantity_error < 1e-6
    for split in ["train", "val"]:
        old = json.loads((Path(config.base) / f"{split}.json").read_text())
        new = json.loads((output / f"{split}.json").read_text())
        assert [record for record in old if record["task"] != "sidestep"] == [
            record for record in new if record["task"] != "sidestep"
        ]
    training = MotionDataset(
        output / "train.json", "train", ["sidestep"], config.project
    )
    validation = MotionDataset(output / "val.json", "val", ["sidestep"], config.project)
    assert_disjoint(training, validation)
    groups = {}
    for record in records:
        if record["split"] == "train":
            groups.setdefault(record["family"], []).append(record)
    randomizer = random.Random(config.review_seed)
    families = sorted(groups)
    randomizer.shuffle(families)
    assert len(families) >= 10
    selected = [
        randomizer.choice(sorted(groups[family], key=lambda record: record["key"]))
        for family in families[:10]
    ]
    items = []
    for number, record in enumerate(selected, 1):
        pair = []
        for name in ["native_left_cache", "cache"]:
            with np.load(record[name]) as archive:
                with torch.no_grad():
                    pair.append(
                        skeleton(torch.from_numpy(archive["motion"])[None])[0].numpy()
                    )
        video = review / f"{number:02d}_sidestep.mp4"
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
                "1200x760",
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
                "20",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(video),
            ],
            stdin=subprocess.PIPE,
        )
        for frame in range(len(pair[0])):
            encoder.stdin.write(
                draw_frame(
                    pair, skeleton.parents + [20, 21], frame, number, record
                ).tobytes()
            )
        encoder.stdin.close()
        assert encoder.wait() == 0
        contact = Image.new("RGB", (1200, 1520), "white")
        for row, frame in enumerate(
            [0, len(pair[0]) // 3, 2 * len(pair[0]) // 3, len(pair[0]) - 1]
        ):
            contact.paste(
                draw_frame(
                    pair, skeleton.parents + [20, 21], frame, number, record
                ).resize((600, 760 // 2)),
                ((row % 2) * 600, (row // 2) * 380),
            )
        contact = contact.crop((0, 0, 1200, 760))
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
            int(probe["nb_frames"]) == len(pair[0]) and probe["r_frame_rate"] == "20/1"
        )
        items.append(
            dict(
                number=number,
                video=video.name,
                sha256=file_sha256(video),
                frames=len(pair[0]),
                record=record,
            )
        )
        print(video.name, flush=True)
    save_json(
        review / "manifest.json",
        dict(
            seed=config.review_seed,
            sampling="One seeded random cycle from each of10 distinct TRAIN source families; paired original-left and mirrored-right. Native-left panels are references, right panels are exact training caches.",
            items=items,
        ),
    )
    verification = dict(
        cache_pairs=len(records),
        cache_count=2 * len(records),
        exact_source_crops=True,
        quantity_max_error=quantity_error,
        physical_rules_pass=True,
        other19_tasks_unchanged=True,
        loader_counts=dict(train=len(training.rows), val=len(validation.rows)),
        source_family_splits_disjoint=True,
        review_distinct_train_families=len({record["family"] for record in selected}),
        videos=10,
        fps=20,
        video_frames_verified=True,
        user_review="pending",
        training_started=False,
    )
    save_json(review / "verification.json", verification)
    links = "".join(
        f'<article><h2>#{item["number"]:02d} · {item["record"]["quantity"]:.3f} m</h2><p>{html.escape(item["record"]["family"])}</p><video controls loop preload="metadata" src="{item["video"]}"></video></article>'
        for item in items
    )
    (review / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>HDM05 sidestep 来源复核</title><style>body{font:16px system-ui;max-width:1220px;margin:30px auto;background:#f3f6f9;color:#17324a}article{background:white;padding:20px;margin:20px 0}video{width:100%}p{line-height:1.7}</style><h1>sidestep：10条训练样本与原始左侧对照</h1><p>每条来自不同TRAIN录制。左面板是HDM05原始左侧步参考，右面板是镜像后实际训练缓存；红色为人体左侧、蓝色为右侧。现有sidestep指标仅测右侧位移，原始左侧未混入当前训练清单。</p><p>仅选01-01场景第6阶段非交叉左侧步，每条截取一次迈开—并拢。官方cuts映射下载403，边界由官方动作脚本和实际运动补标，并保留BABEL act_cat旁证；不宣称是官方walkLeft2/3Steps原始裁剪边界。镜像在SMPL姿态和根平移完成后重新计算FK，固定骨架轻微不对称，因此不是逐点完美几何反射。</p><p>34条train／6条val（10／2录制）。目前未训练；其他任务暂缓。<a href="manifest.json">逐条来源</a> · <a href="verification.json">核验</a> · <a href="coverage/report.md">参数覆盖</a> · <a href="audit.json">筛选统计</a></p>'
        + links
    )
    print(json.dumps(verification), flush=True)


if __name__ == "__main__":
    main()
