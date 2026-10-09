"""Verify kick caches, select amplitude-stratified TRAIN samples and render raw FK."""

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

from data_processing.kick.repair import physical_check
from shared_motion.training.catalog import measure
from shared_motion.training.geometry import Skeleton
from shared_motion.training.model import file_sha256
from shared_motion.training.runner import save_json

LEFT_JOINTS = {1, 4, 7, 10, 13, 16, 18, 20, 22}
RIGHT_JOINTS = {2, 5, 8, 11, 14, 17, 19, 21, 23}


def draw_frame(joints, parents, frame, number, record):
    canvas = Image.new("RGB", (1000, 660), "#f5f8fb")
    drawing = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 19)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 14)
    drawing.text(
        (18, 12),
        f"#{number:02d} RIGHT FORWARD KICK | measured command {record['quantity']:.3f} m",
        fill="#183248",
        font=font,
    )
    drawing.text(
        (18, 40),
        f"{record['family']} | {record['caption']}",
        fill="#183248",
        font=small,
    )
    drawing.text(
        (18, 62),
        f"Source {record['crop_start_frame_20fps']/20:.2f}-{record['crop_end_frame_20fps']/20:.2f}s | t={frame/20:.2f}s | frame {frame+1}/{len(joints)} | {record['annotation_level']}",
        fill="#183248",
        font=small,
    )
    drawing.text(
        (18, 84),
        "Anatomical left: red / right: blue. Fixed cameras, unchanged source motion, 20 fps.",
        fill="#536477",
        font=small,
    )
    center = joints[:, 0].mean(0)
    projections = [
        np.array([[0, -1, 0], [0, 0, -1]]),
        np.array([[0.707, -0.707, 0], [0.354, 0.354, -0.866]]),
    ]
    floor_height = float(joints[:, [7, 8, 10, 11], 2].min())
    for column, projection in enumerate(projections):
        projected = (joints - center) @ projection.T
        extent = np.ptp(projected, axis=(0, 1))
        scale = min(235, 450 / max(extent[0], 0.1), 390 / max(extent[1], 0.1))
        midpoint = (projected.min((0, 1)) + projected.max((0, 1))) / 2
        origin = np.array([250 + column * 500, 335]) - midpoint * scale

        def project(points):
            return [
                tuple(pair)
                for pair in (
                    (np.asarray(points) - center) @ projection.T * scale + origin
                ).reshape(-1, 2)
            ]

        drawing.text(
            (18 + column * 500, 111),
            ["FRONT", "OBLIQUE"][column],
            fill="#536477",
            font=small,
        )
        for axis in [0, 1]:
            for offset in np.arange(-1, 1.1, 0.25):
                first = np.array([-0.9, -0.9, floor_height])
                last = np.array([0.9, 0.9, floor_height])
                first[axis] = last[axis] = offset
                drawing.line(project([first, last]), fill="#dde5ed", width=1)
        for joint_index in range(1, 24):
            color = (
                "#d54b42"
                if joint_index in LEFT_JOINTS
                else "#2375ba" if joint_index in RIGHT_JOINTS else "#354554"
            )
            drawing.line(
                project(joints[frame, [parents[joint_index], joint_index]]),
                fill=color,
                width=5,
            )
        for joint_index in [15, 7, 8]:
            horizontal, vertical = project(joints[frame, [joint_index]])[0]
            drawing.ellipse(
                (horizontal - 4, vertical - 4, horizontal + 4, vertical + 4),
                fill=(
                    "#2375ba"
                    if joint_index == 8
                    else "#d54b42" if joint_index == 7 else "#354554"
                ),
            )
    drawing.rectangle((0, 545, 1000, 660), fill="white")
    drawing.text(
        (18, 560),
        f"Forward excursion: {record['metrics']['forward_excursion_m']:.3f} m | right ankle lift: {record['metrics']['ankle_lift_m']:.3f} m",
        fill="#183248",
        font=font,
    )
    drawing.text(
        (18, 591),
        f"Pelvis excursion: {record['metrics']['root_excursion_m']:.3f} m | support foot excursion: {record['metrics']['support_foot_excursion_m']:.3f} m",
        fill="#183248",
        font=font,
    )
    drawing.text(
        (18, 624),
        "Actual TRAIN cache. No mirroring, retiming, stabilization or generated motion. Human review pending.",
        fill="#536477",
        font=small,
    )
    return canvas


@hydra.main(version_base="1.3", config_path="../../config", config_name="kick_review")
def main(config):
    data = Path(config.data)
    output = data / "review"
    output.mkdir(exist_ok=False)
    policy = OmegaConf.load(data / "config.yaml")
    records = json.loads((data / "kick_index.json").read_text())
    audit = json.loads((data / "audit.json").read_text())
    skeleton_path = (
        Path(config.project)
        / "outputs_amass/transfer_charlie_20261008/snapshot/outputs_amass/franken_eleven_20261003/skeleton.npz"
    )
    skeleton = Skeleton(skeleton_path)
    parents = skeleton.parents + [20, 21]
    torch.set_num_threads(2)
    for split in ["train", "val"]:
        old = json.loads((Path(audit["base"]) / f"{split}.json").read_text())
        new = json.loads((data / f"{split}.json").read_text())
        assert [record for record in old if record["task"] != "kick"] == [
            record for record in new if record["task"] != "kick"
        ]
    maximum_error = 0
    for record in records:
        assert file_sha256(record["cache"]) == record["cache_sha256"]
        with np.load(record["cache"]) as cache:
            motion = cache["motion"].copy()
            source = np.load(record["motion_source"], mmap_mode="r")
            assert np.array_equal(
                motion,
                source[
                    record["crop_start_frame_20fps"] : record["crop_end_frame_20fps"]
                ],
            )
            assert (
                len(motion) == record["real_frames"] == record["target_frames"]
                and record["pad_frames"] == 0
            )
            assert (
                cache["local"].shape == (len(motion), 408) and int(cache["task"]) == 7
            )
            assert not cache["local"][~cache["local_mask"]].any()
            with torch.no_grad():
                joints = skeleton(torch.from_numpy(motion)[None])[0].numpy()
                quantity = float(
                    measure(
                        skeleton,
                        torch.from_numpy(motion)[None],
                        torch.tensor([7]),
                        torch.tensor([len(motion)]),
                    )[0]
                )
            _, failures = physical_check(
                joints, record["local_peak_frame"], policy.rules
            )
            assert not failures, (record["key"], failures)
            maximum_error = max(
                maximum_error,
                abs(quantity - record["quantity"]),
                abs(quantity - float(cache["quantity"])),
            )
    assert maximum_error < 1e-6
    training = sorted(
        [record for record in records if record["split"] == "train"],
        key=lambda record: (record["quantity"], record["key"]),
    )
    assert len(training) >= 10
    randomizer = random.Random(config.seed)
    selected = []
    used_families = set()
    for band in np.array_split(np.arange(len(training)), 10):
        candidates = [training[int(index)] for index in band]
        randomizer.shuffle(candidates)
        record = next(
            (
                candidate
                for candidate in candidates
                if candidate["family"] not in used_families
            ),
            candidates[0],
        )
        selected.append(record)
        used_families.add(record["family"])
    assert len({record["key"] for record in selected}) == 10
    items = []
    for number, record in enumerate(selected, 1):
        motion = np.load(record["cache"])["motion"]
        with torch.no_grad():
            joints = skeleton(torch.from_numpy(motion)[None])[0].numpy()
        video = output / f"{number:02d}_kick.mp4"
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
                "1000x660",
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
        for frame in range(len(joints)):
            encoder.stdin.write(
                draw_frame(joints, parents, frame, number, record).tobytes()
            )
        encoder.stdin.close()
        assert encoder.wait() == 0
        contact = Image.new("RGB", (1500, 660), "white")
        peak_frame = record["local_peak_frame"]
        for contact_index, frame in enumerate(
            [
                0,
                peak_frame - 6,
                peak_frame - 3,
                peak_frame,
                min(len(joints) - 1, peak_frame + 5),
                len(joints) - 1,
            ]
        ):
            contact.paste(
                draw_frame(joints, parents, frame, number, record).resize((500, 330)),
                ((contact_index % 3) * 500, (contact_index // 3) * 330),
            )
        contact.save(output / f"{number:02d}_contact.jpg", quality=93)
        draw_frame(joints, parents, peak_frame, number, record).save(
            output / f"{number:02d}_poster.jpg", quality=92
        )
        probe = json.loads(
            subprocess.check_output(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-count_frames",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=nb_read_frames,r_frame_rate",
                    "-of",
                    "json",
                    str(video),
                ]
            )
        )["streams"][0]
        assert (
            int(probe["nb_read_frames"]) == len(joints)
            and probe["r_frame_rate"] == "20/1"
        )
        items.append(
            dict(
                number=number,
                video=video.name,
                frames=len(joints),
                sha256=file_sha256(video),
                row=record,
            )
        )
        print(video.name, flush=True)
    articles = []
    for item in items:
        record = item["row"]
        articles.append(
            f'<article><h2>#{item["number"]:02d} · {record["quantity"]:.3f} m</h2><p>{html.escape(record["family"])}</p><p>{html.escape(record["caption"])} · {record["annotation_level"]} · 源时间 {record["crop_start_frame_20fps"]/20:.2f}–{record["crop_end_frame_20fps"]/20:.2f}s</p><video controls loop preload="metadata" poster="{item["number"]:02d}_poster.jpg" src="{item["video"]}"></video></article>'
        )
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>kick 训练样本复核</title><style>body{font:16px system-ui;max-width:1060px;margin:35px auto;padding:18px;background:#f3f6fa}article{background:white;padding:20px;margin:25px 0;border-radius:10px}video{width:100%}</style><h1>kick：10 条真实训练样本</h1><p>按训练幅度排序分成10个数量相近的组，每组固定seed抽1条，尽量来自不同原始录制。样本并非按目标范围重新缩放。右腿蓝色，左腿红色；保留原始运动。尚待用户复核。</p><p>frame为BABEL帧级动作；single_action_sequence为BABEL明确标记的单动作序列，并经过实际动作事件定位与准入。类别标签不代表参数覆盖已充分。</p><p><a href="manifest.json">来源与裁剪清单</a> · <a href="verification.json">核验</a></p>'
        + "".join(articles)
    )
    save_json(
        output / "manifest.json",
        dict(
            seed=config.seed,
            sampling="Ten equal-count amplitude bands from TRAIN; one seeded sample each, prefer distinct families. No quantity alteration.",
            items=items,
        ),
    )
    save_json(
        output / "verification.json",
        dict(
            verified=True,
            cache_count=len(records),
            quantity_max_error=maximum_error,
            exact_source_crops=True,
            other19_tasks_unchanged=True,
            videos=10,
            distinct_training_families=len(used_families),
            video_frames_and_fps_verified=True,
            user_review="pending",
        ),
    )


if __name__ == "__main__":
    main()
