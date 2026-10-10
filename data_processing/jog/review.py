"""Render diagnostic examples of existing jog data; never modify motion."""

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
        f'#{number} CURRENT JOG DATA | {reason} | {record["quantity"]:.3f} m/s',
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
        trace = projected[:, 0] * scale + origin
        drawing.line([tuple(point) for point in trace], fill="#adbaca", width=1)
    categories = sorted(
        {
            category
            for event in entry["overlapping_babel_events"]
            for category in event["label"].get("act_cat") or []
        }
    )
    drawing.text(
        (16, 610),
        f'BABEL categories in clip: {", ".join(categories)[:115] or "no frame categories"}',
        font=small,
        fill="#526477",
    )
    drawing.text(
        (16, 640),
        f'run/jog coverage {entry["coverage"]["run_jog"]:.0%} | walk {entry["coverage"]["walk"]:.0%} | 20fps, no stabilization or cleanup',
        font=small,
        fill="#526477",
    )
    return image


@hydra.main(version_base="1.3", config_path="../../config", config_name="audit_jog")
def main(config):
    torch.set_num_threads(2)
    output = Path(config.output)
    rows = json.loads((output / "current_jog_audit.json").read_text())
    training = [entry for entry in rows if entry["record"]["split"] == "train"]
    selected = []
    families = set()
    criteria = [
        (
            "VERY LOW ROOT SPEED",
            lambda entry: entry["flags"]["little_root_travel"],
            lambda entry: entry["metrics"]["path_speed_m_s"],
        ),
        (
            "WALK LABEL DOMINATES",
            lambda entry: entry["flags"]["walk_without_run_for_majority"],
            lambda entry: -entry["coverage"]["walk"],
        ),
        (
            "OTHER ACTION / MIXED CLIP",
            lambda entry: entry["flags"]["incompatible_without_run_at_least_quarter"],
            lambda entry: -entry["coverage"]["incompatible"],
        ),
        (
            "MISSED RUN/JOG EVENT",
            lambda entry: entry["flags"]["frame_run_jog_exists_but_crop_misses"],
            lambda entry: entry["metrics"]["path_speed_m_s"],
        ),
        (
            "BACKWARD MOTION",
            lambda entry: entry["flags"]["backward_motion"]
            and entry["metrics"]["path_speed_m_s"] > 0.8,
            lambda entry: entry["metrics"]["forward_speed_m_s"],
        ),
        (
            "RUN/JOG REFERENCE",
            lambda entry: entry["coverage"]["run_jog"] >= 0.8
            and 0.8 <= entry["record"]["quantity"] <= 2
            and entry["metrics"]["near_stationary_fraction"] < 0.1
            and entry["metrics"]["backward_motion_fraction"] < 0.1,
            lambda entry: (
                not entry["record"]["family"].startswith("MPI_HDM05/"),
                -entry["coverage"]["run_jog"],
            ),
        ),
    ]
    for reason, predicate, sort_key in criteria:
        choices = [
            entry
            for entry in training
            if entry["record"]["family"] not in families and predicate(entry)
        ]
        if not choices:
            continue
        entry = min(choices, key=sort_key)
        selected.append((reason, entry))
        families.add(entry["record"]["family"])
    review = output / "review"
    review.mkdir(exist_ok=True)
    skeleton = Skeleton(config.skeleton)
    items = []
    for number, (reason, entry) in enumerate(selected, 1):
        record = entry["record"]
        with np.load(record["cache"]) as archive:
            motion = archive["motion"][: record["real_frames"]].copy()
        with torch.no_grad():
            joints = skeleton(torch.from_numpy(motion)[None])[0].numpy()
        video = review / f"{number:02d}_jog_diagnostic.mp4"
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
                "1100x680",
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
                    joints, skeleton.parents + [20, 21], frame, entry, number, reason
                ).tobytes()
            )
        encoder.stdin.close()
        assert encoder.wait() == 0
        keyframes = np.linspace(0, len(joints) - 1, 5, dtype=int).tolist()
        contact = Image.new("RGB", (2200, 680), "white")
        for column, frame in enumerate(keyframes[:4]):
            contact.paste(
                draw_frame(
                    joints, skeleton.parents + [20, 21], frame, entry, number, reason
                ).resize((550, 340)),
                (column * 550, 0),
            )
        contact.paste(
            draw_frame(
                joints,
                skeleton.parents + [20, 21],
                keyframes[-1],
                entry,
                number,
                reason,
            ).resize((550, 340)),
            (0, 340),
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
                reason=reason,
                video=video.name,
                video_sha256=file_sha256(video),
                keyframes=keyframes,
                audit=entry,
            )
        )
    save_json(
        review / "manifest.json",
        dict(
            purpose="Targeted diagnostics, not a random quality estimate or cleaned training set",
            items=items,
        ),
    )
    sections = "".join(
        f'<article><h2>#{item["number"]} · {html.escape(item["reason"])}</h2><p>{html.escape(item["audit"]["record"]["family"])}</p><p>{html.escape(item["audit"]["record"]["caption"])}</p><video controls loop preload="metadata" src="{item["video"]}"></video></article>'
        for item in items
    )
    (review / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>jog现有数据问题检查</title><style>body{font:16px system-ui;max-width:1100px;margin:28px auto;background:#f3f6f9}article{padding:20px;background:white;margin:20px 0}video{width:100%}</style><h1>jog：现有数据诊断样本</h1><p>针对低位移、走路、其他动作、错裁、倒跑选取问题样本，另附跑步参考。不是清理后的数据，不表示抽样错误率。全部原缓存直接FK，20fps；未改变训练清单，未训练。</p><p><a href="manifest.json">来源、时段与完整证据</a></p>'
        + sections
    )
    print(json.dumps(dict(videos=len(items), review=str(review))), flush=True)


if __name__ == "__main__":
    main()
