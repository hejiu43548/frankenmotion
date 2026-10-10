"""Render ten distinct TRAIN sources for each repaired running task."""

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
        f'#{number} {record["task"].upper()} | {reason} | control {record["quantity"]:.3f}',
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
    metrics = record["metrics"]
    drawing.text(
        (16, 610),
        f'Root speed {metrics["mean_path_speed_m_s"]:.3f} m/s | excursion {metrics["root_excursion_m"]:.3f} m | steps {metrics["step_rate_hz"]:.2f} /s',
        font=small,
        fill="#526477",
    )
    drawing.text(
        (16, 640),
        f'Alternation {metrics["alternation_fraction"]:.0%} | flight proxy {metrics["flight_fraction"]:.0%} | unchanged source crop / 20fps / not trained',
        font=small,
        fill="#526477",
    )
    return image


def select_examples(records, task):
    candidates = [
        record
        for record in records
        if record["task"] == task and record["split"] == "train"
    ]
    if len(candidates) <= 10:
        return sorted(
            candidates, key=lambda record: (record["quantity"], record["key"])
        )
    target_count = min(10, len({record["family"] for record in candidates}))
    selected = []
    families = set()
    for record in candidates:
        if (
            record["family"].startswith("MPI_HDM05/")
            and record["family"] not in families
        ):
            selected.append(record)
            families.add(record["family"])
            if len(selected) == 2:
                break
    targets = np.linspace(
        min(record["quantity"] for record in candidates),
        max(record["quantity"] for record in candidates),
        target_count - len(selected),
    )
    for target in targets:
        record = min(
            [record for record in candidates if record["family"] not in families],
            key=lambda record: (abs(record["quantity"] - target), record["key"]),
        )
        selected.append(record)
        families.add(record["family"])
    assert len(selected) == len(families) == target_count
    return selected


@hydra.main(
    version_base="1.3", config_path="../../config", config_name="jog_march_repair"
)
def main(config):
    torch.set_num_threads(2)
    output = Path(config.output)
    finalized = (output / "running_index.json").is_file()
    records = json.loads(
        (
            output / ("running_index.json" if finalized else "inspection.json")
        ).read_text()
    )
    review = output / ("review" if finalized else "preview")
    review.mkdir(exist_ok=True)
    skeleton = Skeleton(config.skeleton)
    all_items = []
    for task in ["jog", "march"]:
        folder = review / task
        folder.mkdir(exist_ok=True)
        items = []
        selected = select_examples(records, task)
        for number, record in enumerate(selected, 1):
            motion = np.load(record["motion_source"], mmap_mode="r")[
                record["crop_start_frame_20fps"] : record["crop_end_frame_20fps"]
            ].copy()
            if finalized:
                with np.load(record["cache"]) as cache:
                    assert np.array_equal(cache["motion"], motion)
            with torch.no_grad():
                joints = skeleton(torch.from_numpy(motion)[None])[0].numpy()
            video = folder / f"{number:02d}_{task}.mp4"
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
            reason = (
                (
                    "STRAIGHT RUNNING / m/s"
                    if config.get("straight")
                    else "FORWARD RUNNING / m/s"
                )
                if task == "jog"
                else "IN-PLACE RUNNING / ankle lift m"
            )
            entry = dict(record=record)
            for frame in range(len(joints)):
                encoder.stdin.write(
                    draw_frame(
                        joints,
                        skeleton.parents + [20, 21],
                        frame,
                        entry,
                        number,
                        reason,
                    ).tobytes()
                )
            encoder.stdin.close()
            assert encoder.wait() == 0
            keyframes = np.linspace(0, len(joints) - 1, 6, dtype=int).tolist()
            contact = Image.new("RGB", (1650, 680), "white")
            for position, frame in enumerate(keyframes):
                contact.paste(
                    draw_frame(
                        joints,
                        skeleton.parents + [20, 21],
                        frame,
                        entry,
                        number,
                        reason,
                    ).resize((550, 340)),
                    ((position % 3) * 550, (position // 3) * 340),
                )
            contact.save(folder / f"{number:02d}_contact.png")
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
                int(probe["nb_frames"]) == len(joints)
                and probe["r_frame_rate"] == "20/1"
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
        title = (
            ("jog：直线跑步" if config.get("straight") else "jog：向前跑步")
            if task == "jog"
            else "march：原地跑步"
        )
        family_count = len({item["record"]["family"] for item in items})
        counts = {
            split: sum(
                record["task"] == task and record["split"] == split
                for record in records
            )
            for split in ["train", "val"]
        }
        sections = "".join(
            f'<article><h2>#{item["number"]:02d} · {item["record"]["quantity"]:.3f} {"m/s" if task=="jog" else "m"}</h2><p>{html.escape(item["record"]["family"])}</p><p>{html.escape(item["record"]["original_caption"])} · {item["record"]["annotation_level"]}</p><video controls loop preload="metadata" src="{item["video"]}"></video></article>'
            for item in items
        )
        (folder / "index.html").write_text(
            f'<!doctype html><meta charset="utf-8"><title>{title}</title><style>body{{font:16px system-ui;max-width:1100px;margin:28px auto;background:#f3f6f9}}article{{padding:20px;background:white;margin:20px 0}}video{{width:100%}}</style><h1>{title}</h1><p>候选 {counts["train"]} train / {counts["val"]} val；{len(items)}条TRAIN样本，来自{family_count}个录制；不足10条时全部展示，不重复凑数。仅真实裁剪，无重定时、补帧或轨迹修正。未训练，待用户复核。</p><p><a href="../index.html">两个任务总览</a> · <a href="manifest.json">逐条来源和准入指标</a></p>'
            + sections
        )
        save_json(
            folder / "manifest.json",
            dict(
                task=task,
                counts=counts,
                selection="All TRAIN records if at most10; otherwise up to10 distinct families, HDM05 priority then quantity stratification",
                displayed_samples=len(items),
                distinct_families=family_count,
                finalized=finalized,
                items=items,
            ),
        )
        all_items.extend(items)
    (review / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>jog / march联合修复</title><style>body{font:18px system-ui;max-width:900px;margin:50px auto}a{display:block;padding:24px;background:#eef4f8;margin:16px 0}</style><h1>跑步数据分流复核</h1><a href="jog/index.html">jog · 跑步训练样本</a><a href="march/index.html">march · 原地跑步训练样本</a><p>先核实是跑步，再按实际根位移分流。站立、走路、假跑和明显脚滑不会仅因低位移而进入march。未训练。</p>'
    )
    save_json(
        review / "verification.json",
        dict(
            videos=len(all_items),
            all_displayed_samples_are_train=True,
            frame_counts_and_20fps_verified=True,
            finalized=finalized,
            training_started=False,
        ),
    )
    print(json.dumps(dict(review=str(review), videos=len(all_items))), flush=True)


if __name__ == "__main__":
    main()
