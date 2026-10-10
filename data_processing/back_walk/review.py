"""Verify repaired backward caches and render10 unchanged TRAIN clips."""

import html
import json
from pathlib import Path
import random
import subprocess

import hydra
import numpy as np
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import torch

from data_processing.back_walk.repair import check_clip
from shared_motion.training.catalog import measure
from shared_motion.training.data import MotionDataset
from shared_motion.training.data import assert_disjoint
from shared_motion.training.geometry import Skeleton
from shared_motion.training.model import file_sha256
from shared_motion.training.runner import save_json


def draw_frame(joints, parents, frame, record, number):
    image = Image.new("RGB", (1000, 650), "#f5f8fb")
    drawing = ImageDraw.Draw(image)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 18)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    drawing.text(
        (16, 12),
        f'#{number:02d} BACKWARD WALK | TRAIN | speed {record["quantity"]:.3f} m/s',
        font=font,
        fill="#17324a",
    )
    drawing.text((16, 40), record["family"], font=small, fill="#17324a")
    drawing.text(
        (16, 63),
        f'Source {record["crop_start_frame_20fps"]/20:.2f}-{record["crop_end_frame_20fps"]/20:.2f}s | t={frame/20:.2f}s | {frame+1}/{len(joints)} frames',
        font=small,
        fill="#536477",
    )
    projections = [
        np.array([[1, 0, 0], [0, 0, -1]]),
        np.array([[0.707, -0.707, 0], [0.354, 0.354, -0.866]]),
    ]
    left_joints = {1, 4, 7, 10, 13, 16, 18, 20, 22}
    right_joints = {2, 5, 8, 11, 14, 17, 19, 21, 23}
    for column, projection in enumerate(projections):
        projected = joints @ projection.T
        center = (projected.min((0, 1)) + projected.max((0, 1))) / 2
        scale = min(235, 435 / max(np.ptp(projected, axis=(0, 1)).max(), 0.1))
        positions = (projected[frame] - center) * scale + np.array(
            [column * 500 + 250, 345]
        )
        drawing.text(
            (column * 500 + 16, 99),
            ["SIDE VIEW | forward +X ->", "OBLIQUE VIEW"][column],
            font=small,
            fill="#536477",
        )
        for joint in range(1, 24):
            color = (
                "#d54b42"
                if joint in left_joints
                else "#2375ba" if joint in right_joints else "#354554"
            )
            drawing.line(
                [tuple(positions[parents[joint]]), tuple(positions[joint])],
                fill=color,
                width=4,
            )
        trace = (projected[:, 0] - center) * scale + np.array([column * 500 + 250, 345])
        drawing.line([tuple(point) for point in trace], fill="#c0ccd7", width=1)
    drawing.text(
        (16, 592),
        "Raw FK / fixed cameras /20fps / red=left, blue=right / no padding or time reversal",
        font=small,
        fill="#536477",
    )
    drawing.text(
        (16, 617),
        f'Forward path fraction {record["metrics"]["forward_path_fraction"]:.3%} | Original caption retained in audit JSON',
        font=small,
        fill="#536477",
    )
    return image


@hydra.main(
    version_base="1.3", config_path="../../config", config_name="back_walk_repair"
)
def main(config):
    output = Path(config.output)
    review = output / "review"
    review.mkdir(exist_ok=False)
    records = json.loads((output / "back_walk_index.json").read_text())
    skeleton = Skeleton(config.skeleton)
    torch.set_num_threads(2)
    maximum_error = 0.0
    for record in records:
        with np.load(record["cache"]) as cache:
            motion = cache["motion"].copy()
            assert file_sha256(record["cache"]) == record["cache_sha256"]
            if record["source_mode"] == "full_original_annotation":
                source = np.load(record["motion_source"], mmap_mode="r")
            else:
                with np.load(record["motion_source"]) as original:
                    source = original["motion"].copy()
            assert np.array_equal(
                motion,
                source[record["source_slice_start"] : record["source_slice_stop"]],
            )
            with torch.no_grad():
                joints = skeleton(torch.from_numpy(motion)[None])[0].numpy()
                value = float(
                    measure(
                        skeleton,
                        torch.from_numpy(motion)[None],
                        torch.tensor([6]),
                        torch.tensor([len(motion)]),
                    )[0]
                )
            assert not check_clip(joints, config.rules)[1]
            maximum_error = max(maximum_error, abs(value - float(cache["quantity"])))
            assert len(motion) == record["real_frames"] and record["pad_frames"] == 0
    assert maximum_error < 1e-6
    for split in ["train", "val"]:
        before = json.loads((Path(config.base) / f"{split}.json").read_text())
        after = json.loads((output / f"{split}.json").read_text())
        assert [record for record in before if record["task"] != "back_walk"] == [
            record for record in after if record["task"] != "back_walk"
        ]
    training = MotionDataset(
        output / "train.json", "train", ["back_walk"], config.project
    )
    validation = MotionDataset(
        output / "val.json", "val", ["back_walk"], config.project
    )
    assert_disjoint(training, validation)
    candidates = [record for record in records if record["split"] == "train"]
    random.Random(config.seed).shuffle(candidates)
    selected = []
    families = set()
    for record in candidates:
        if record["family"] not in families:
            selected.append(record)
            families.add(record["family"])
        if len(selected) == 10:
            break
    assert len(selected) == 10
    items = []
    for number, record in enumerate(selected, 1):
        with np.load(record["cache"]) as cache:
            with torch.no_grad():
                joints = skeleton(torch.from_numpy(cache["motion"])[None])[0].numpy()
        video = review / f"{number:02d}_back_walk.mp4"
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
        contact = Image.new("RGB", (1500, 325), "white")
        for column, frame in enumerate([0, len(joints) // 2, len(joints) - 1]):
            contact.paste(
                draw_frame(
                    joints, skeleton.parents + [20, 21], frame, record, number
                ).resize((500, 325)),
                (column * 500, 0),
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
                number=number,
                video=video.name,
                sha256=file_sha256(video),
                frames=len(joints),
                record=record,
            )
        )
    save_json(
        review / "manifest.json",
        dict(
            seed=config.seed,
            sampling="Seeded shuffle, first10 distinct TRAIN families; no generated motion",
            items=items,
        ),
    )
    save_json(
        review / "verification.json",
        dict(
            cache_count=len(records),
            source_crops_exact=True,
            quantity_max_error=maximum_error,
            other19_tasks_unchanged=True,
            split_isolation=True,
            loader_counts=dict(train=len(training.rows), val=len(validation.rows)),
            videos=10,
            fps=20,
            video_frames_verified=True,
            user_review="pending",
        ),
    )
    sections = "".join(
        f'<article><h2>#{entry["number"]:02d} · {entry["record"]["quantity"]:.3f}m/s</h2><p>{html.escape(entry["record"]["family"])}</p><video controls loop preload="metadata" src="{entry["video"]}"></video></article>'
        for entry in items
    )
    (review / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>back_walk来源复核</title><style>body{font:16px system-ui;max-width:1050px;margin:28px auto;background:#f3f6f9}article{padding:20px;background:white;margin:20px 0}video{width:100%}</style><h1>back_walk：10条实际训练样本</h1><p>仅裁取现有来源的连续倒走段；纯前走/无有效倒走段剔除。速度重算、无补帧、未倒放。左侧视图朝向+X为向前，倒走应往左；红=人体左侧，蓝=右侧。其余19类记录不变，未启动训练。</p><p><a href="manifest.json">原始描述、裁剪前后与新标签</a> · <a href="verification.json">核验</a> · <a href="../audits_back_walk/back_walk/index.html">逐条训练/验证列表</a></p>'
        + sections
    )
    print(
        json.dumps(
            dict(
                videos=10,
                verified_caches=len(records),
                quantity_max_error=maximum_error,
            )
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
