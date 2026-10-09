"""Verify repaired caches and render a deterministic ten-source training review."""

import argparse
from collections import Counter
import html
import json
from pathlib import Path
import random
import subprocess
import sys
import textwrap
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from shared_motion.training.catalog import measure
from shared_motion.training.geometry import Skeleton
from shared_motion.training.model import file_sha256 as sha
from shared_motion.training.runner import save_json

LEFT = {1, 4, 7, 10, 13, 16, 18, 20, 22}
RIGHT = {2, 5, 8, 11, 14, 17, 19, 21, 23}


def frame_image(joints, parents, frame, number, row):
    canvas = Image.new("RGB", (1000, 720), "#f7f9fb")
    d = ImageDraw.Draw(canvas)
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 20)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    d.text(
        (20, 12),
        f"#{number:02d}  WAVE TRAINING SOURCE  |  {row['source_kind']}",
        font=font,
        fill="#15283a",
    )
    d.text(
        (20, 42),
        f"{Path(row['family']).name}  |  {row['caption']}",
        font=small,
        fill="#15283a",
    )
    d.text(
        (20, 65),
        f"Source {row['crop_start_frame_20fps']/20:.2f}-{row['crop_end_frame_20fps']/20:.2f}s | t={frame/20:.2f}s | frame {frame+1}/{len(joints)} | measured command={row['quantity']:.3f}m",
        font=small,
        fill="#15283a",
    )
    d.text(
        (20, 90),
        "Anatomical LEFT = red | RIGHT = blue. Fixed cameras and ground; original motion, no stabilization.",
        font=small,
        fill="#536477",
    )
    projections = [
        np.array([[0, -1, 0], [0, 0, -1]]),
        np.array([[0.707, -0.707, 0], [0.354, 0.354, -0.866]]),
    ]
    floor = float(joints[:, [7, 8, 10, 11], 2].min())
    # A fixed per-clip framing transform does not remove any motion or drift.
    center = joints[:, 0].mean(0)
    center[2] = (joints[:, :, 2].min() + joints[:, :, 2].max()) / 2
    for col, projection in enumerate(projections):
        projected = (joints - center) @ projection.T
        extent = np.ptp(projected, axis=(0, 1))
        scale = min(235, 460 / max(extent[0], 0.1), 380 / max(extent[1], 0.1))
        midpoint = (projected.min((0, 1)) + projected.max((0, 1))) / 2
        origin = np.array([250 + 500 * col, 340]) - midpoint * scale

        def p(x):
            return [
                tuple(v)
                for v in (
                    (np.asarray(x) - center) @ projection.T * scale + origin
                ).reshape(-1, 2)
            ]

        d.text(
            (col * 500 + 20, 120), ["FRONT", "OBLIQUE"][col], font=small, fill="#536477"
        )
        for axis in [0, 1]:
            for offset in np.arange(-1, 1.1, 0.25):
                a = np.array([-0.9, -0.9, floor])
                b = np.array([0.9, 0.9, floor])
                a[axis] = b[axis] = offset
                d.line(p([a, b]), fill="#dde4eb", width=1)
        for joint in range(1, 24):
            color = (
                "#d54b42"
                if joint in LEFT
                else "#2375ba" if joint in RIGHT else "#3c4954"
            )
            d.line(p(joints[frame, [parents[joint], joint]]), fill=color, width=5)
        for joint in [15, 20, 21]:
            x, y = p(joints[frame, [joint]])[0]
            color = (
                "#d54b42" if joint == 20 else "#2375ba" if joint == 21 else "#3c4954"
            )
            d.ellipse((x - 5, y - 5, x + 5, y + 5), fill=color)
    d.rectangle((0, 551, 1000, 720), fill="white")
    d.text(
        (20, 565),
        "Foot / pelvis paths (XY): fixed scale, 1 m = 180 px",
        font=small,
        fill="#536477",
    )
    for joint, color in [(0, "#536477"), (7, "#d54b42"), (8, "#2375ba")]:
        xy = joints[:, joint, :2] * np.array([1, -1]) * 180 + np.array([245, 650])
        d.line([tuple(x) for x in xy], fill="#d9dfe5", width=2)
        if frame:
            d.line([tuple(x) for x in xy[: frame + 1]], fill=color, width=3)
        x, y = xy[frame]
        d.ellipse((x - 4, y - 4, x + 4, y + 4), fill=color)
    m = row["metrics"]
    d.text(
        (510, 575),
        f"Pelvis excursion: {m['root_excursion_m']*100:.1f} cm",
        font=font,
        fill="#15283a",
    )
    d.text(
        (510, 605),
        f"Max foot excursion: {m['foot_excursion_m']*100:.1f} cm",
        font=font,
        fill="#15283a",
    )
    d.text(
        (510, 635),
        f"Right-wave reversals: {m['lateral_reversals']}",
        font=font,
        fill="#15283a",
    )
    d.text(
        (510, 675),
        "FK skeleton, 20 fps. Awaiting human review.",
        font=small,
        fill="#536477",
    )
    return canvas


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("data", type=Path)
    args = ap.parse_args()
    data = args.data
    out = data / "review"
    out.mkdir(exist_ok=False)
    torch.set_num_threads(2)
    root = Path("/home/psirobot/projects/frankenmotion")
    skpath = (
        root
        / "outputs_amass/transfer_charlie_20261008/snapshot/outputs_amass/franken_eleven_20261003/skeleton.npz"
    )
    sk = Skeleton(skpath)
    parents = sk.parents + [20, 21]
    rows = json.loads((data / "wave_index.json").read_text())
    audit = json.loads((data / "audit.json").read_text())
    allrows = {
        s: json.loads((data / f"{s}.json").read_text()) for s in ["train", "val"]
    }
    for split in ["train", "val"]:
        old = json.loads((Path(audit["base"]) / f"{split}.json").read_text())
        assert [r for r in old if r["task"] != "wave"] == [
            r for r in allrows[split] if r["task"] != "wave"
        ]
    assert not {r["family"] for r in allrows["train"]} & {
        r["family"] for r in allrows["val"]
    }
    provenance = json.loads((data / "provenance.json").read_text())
    for path, digest in provenance.items():
        assert sha(path) == digest, path
    error = 0
    for row in rows:
        assert sha(row["cache"]) == row["cache_sha256"]
        with np.load(row["cache"]) as z:
            motion = z["motion"].copy()
            n = len(motion)
            assert (
                n == row["real_frames"] == row["target_frames"]
                and row["pad_frames"] == 0
                and 40 <= n <= 120
            )
            full = np.load(row["motion_source"], mmap_mode="r")
            s = row["crop_start_frame_20fps"]
            e = row["crop_end_frame_20fps"]
            assert np.array_equal(motion, full[s:e])
            assert (
                z["local"].shape == (n, 408)
                and z["local_mask"].shape == (n, 408)
                and z["tx"].shape == (512,)
                and int(z["task"]) == 3
            )
            assert (
                np.isfinite(motion).all()
                and np.isfinite(z["local"]).all()
                and np.isfinite(z["tx"]).all()
            )
            assert not z["local"][~z["local_mask"]].any()
            with torch.no_grad():
                q = float(
                    measure(
                        sk,
                        torch.from_numpy(motion)[None],
                        torch.tensor([3]),
                        torch.tensor([n]),
                    )[0]
                )
            error = max(error, abs(q - row["quantity"]), abs(q - float(z["quantity"])))
            assert error < 1e-6
            assert (
                row["source_label"].get("act_cat") is None
                or "wave" in row["source_label"]["act_cat"]
            )
    rng = random.Random(20261009)
    train = [r for r in rows if r["split"] == "train"]
    chosen = []
    used = set()
    for kind, count in [("hdm05_script_supplement", 5), ("babel_act_cat_frame", 5)]:
        pool = sorted(
            [r for r in train if r["source_kind"] == kind], key=lambda r: r["key"]
        )
        rng.shuffle(pool)
        for row in pool:
            if row["family"] in used:
                continue
            chosen.append(row)
            used.add(row["family"])
            if sum(r["source_kind"] == kind for r in chosen) == count:
                break
    if len(chosen) < 10:
        pool = sorted(train, key=lambda r: r["key"])
        rng.shuffle(pool)
        for row in pool:
            if row["family"] not in used:
                chosen.append(row)
                used.add(row["family"])
            if len(chosen) == 10:
                break
    assert len(chosen) == len({r["family"] for r in chosen}) == 10
    media = []
    posters = []
    for number, row in enumerate(chosen, 1):
        motion = np.load(row["cache"])["motion"]
        with torch.no_grad():
            j = sk(torch.from_numpy(motion)[None])[0].numpy()
        video = out / f"{number:02d}_wave.mp4"
        cmd = [
            "ffmpeg",
            "-y",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "rgb24",
            "-s",
            "1000x720",
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
        ]
        process = subprocess.Popen(cmd, stdin=subprocess.PIPE)
        for frame in range(len(j)):
            process.stdin.write(frame_image(j, parents, frame, number, row).tobytes())
        process.stdin.close()
        assert process.wait() == 0
        contact = Image.new("RGB", (1500, 1080), "white")
        for k, frame in enumerate(np.linspace(0, len(j) - 1, 6).astype(int)):
            contact.paste(
                frame_image(j, parents, int(frame), number, row).resize((500, 360)),
                ((k % 3) * 500, (k // 3) * 360),
            )
        # Two rows only; retain full resolution at 1500x720.
        contact = contact.crop((0, 0, 1500, 720))
        contact.save(out / f"{number:02d}_contact.jpg", quality=93)
        poster = frame_image(j, parents, len(j) // 2, number, row)
        poster.save(out / f"{number:02d}_poster.jpg", quality=92)
        posters.append(poster)
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
            int(probe["nb_read_frames"]) == len(j) and probe["r_frame_rate"] == "20/1"
        )
        media.append(
            dict(
                number=number,
                video=video.name,
                sha256=sha(video),
                frames=len(j),
                fps=20,
                row=row,
            )
        )
        print(video.name, flush=True)
    sheet = Image.new("RGB", (2000, 1800), "white")
    for i, poster in enumerate(posters):
        sheet.paste(
            poster.resize((1000, 720)).resize((1000, 360)),
            ((i % 2) * 1000, (i // 2) * 360),
        )
    # Individual six-frame sheets preserve aspect; overview uses compact posters below.
    sheet = Image.new("RGB", (2000, 3600), "white")
    for i, poster in enumerate(posters):
        sheet.paste(poster, ((i % 2) * 1000, (i // 2) * 720))
    sheet.save(out / "overview.jpg", quality=92)
    lines = []
    for entry in media:
        r = entry["row"]
        lines.append(
            f'<article><h2>#{entry["number"]:02d} · {html.escape(r["caption"])}</h2><p>{html.escape(r["source_kind"])} · {html.escape(r["family"])}</p><p>原始时间 {r["crop_start_frame_20fps"]/20:.2f}–{r["crop_end_frame_20fps"]/20:.2f} 秒 · 幅度 {r["quantity"]:.3f} m</p><video controls loop preload="metadata" poster="{entry["number"]:02d}_poster.jpg" src="{entry["video"]}"></video></article>'
        )
    page = (
        '<!doctype html><meta charset="utf-8"><title>wave 训练来源复核 · 10 条</title><style>body{font:16px system-ui;background:#f2f5f8;color:#15283a;max-width:1040px;margin:30px auto;padding:16px}article{background:white;margin:24px 0;padding:20px;border-radius:12px}video{width:100%}p{line-height:1.6}</style><h1>wave：10 条真实训练来源复核</h1><p>固定 seed=20261009，按来源分层抽样，10 个不同原始录制。红色为人体左侧，蓝色为右侧；固定镜头保留身体位移，底部显示双踝和骨盆轨迹。这些是修复后的训练缓存，不是模型生成结果。</p><p>HDM05 时间边界为依据官方脚本和实际动作补充的区间，仍待你复核。其余来自 BABEL 帧级 act_cat=wave。审核通过前不继续下一任务。</p><p><a href="manifest.json">样本来源与裁剪清单</a> · <a href="verification.json">缓存核验</a></p>'
        + "".join(lines)
    )
    (out / "index.html").write_text(page)
    save_json(
        out / "manifest.json",
        dict(
            seed=20261009,
            sampling="5 native HDM05 and5 BABEL, each from a distinct existing TRAIN family; seeded shuffle within source kind; shortages filled from remaining distinct families.",
            items=media,
            render="unchanged training motion FK; fixed cameras; no stabilization or mirroring",
        ),
    )
    save_json(
        out / "verification.json",
        dict(
            verified=True,
            wave_caches_checked=len(rows),
            other19_tasks_exactly_preserved=True,
            source_crops_exact=True,
            quantity_max_error=error,
            split_family_disjoint=True,
            input_hashes_match=True,
            rendered_training_samples=10,
            distinct_training_families=10,
            video_fps_and_frame_count_verified=True,
            human_review="pending",
        ),
    )


if __name__ == "__main__":
    main()
