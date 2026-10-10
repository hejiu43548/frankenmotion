"""Render unmodified saved Stage2 scans, with command and world-trajectory evidence."""

import hashlib
import html
import json
from pathlib import Path
import subprocess
import sys

import hydra
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import torch


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def render_group(output, name, sequences, parents, commands, measured, unit, step):
    """Common fixed camera and scale for all three levels, without motion correction."""
    projection = np.array([[0.707, -0.707, 0], [0.354, 0.354, -0.866]])
    projected = [sequence @ projection.T for sequence in sequences]
    combined = np.concatenate(projected)
    center = (combined.min((0, 1)) + combined.max((0, 1))) / 2
    scale = min(230, 365 / max(float(np.ptp(combined, axis=(0, 1)).max()), 0.1))
    paths = [sequence[:, [0, 7, 8], :2] for sequence in sequences]
    path_bounds = np.concatenate(paths)
    path_center = (path_bounds.min((0, 1)) + path_bounds.max((0, 1))) / 2
    path_scale = min(350, 230 / max(float(np.ptp(path_bounds, axis=(0, 1)).max()), 0.3))
    font_path = "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"
    font = ImageFont.truetype(font_path, 20)
    small = ImageFont.truetype(font_path, 16)
    left_joints = {1, 4, 7, 10, 13, 16, 18, 20, 22}
    right_joints = {2, 5, 8, 11, 14, 17, 19, 21, 23}
    colors = ["#253746", "#d54b42", "#2375ba"]
    frames = max(len(sequence) for sequence in sequences)

    def draw_frame(frame_index, compact=False):
        canvas = Image.new("RGB", (1440, 450 if compact else 820), "#f5f8fb")
        drawing = ImageDraw.Draw(canvas)
        drawing.text(
            (18, 10),
            f"{name} | Stage2 best {step:,} | {frame_index / 20:.2f}s",
            font=font,
            fill="#17324a",
        )
        drawing.text(
            (18, 40),
            "Raw saved FK | fixed common camera | no stabilization or ground correction | red LEFT / blue RIGHT",
            font=small,
            fill="#536477",
        )
        for column, sequence in enumerate(sequences):
            actual_frame = min(frame_index, len(sequence) - 1)
            offset = column * 480
            label = ["LOW", "MID", "HIGH"][column]
            drawing.text(
                (offset + 18, 72),
                f"{label}: request {commands[column]:.3f} {unit}",
                font=font,
                fill="#17324a",
            )
            drawing.text(
                (offset + 18, 101),
                f"Measured {measured[column]:.3f} {unit}",
                font=font,
                fill="#17324a",
            )
            positions = (projected[column][actual_frame] - center) * scale * (
                0.8 if compact else 1.0
            ) + [offset + 240, 280 if compact else 305]
            for joint_index in range(1, 24):
                color = (
                    colors[1]
                    if joint_index in left_joints
                    else colors[2] if joint_index in right_joints else colors[0]
                )
                drawing.line(
                    [
                        tuple(positions[parents[joint_index]]),
                        tuple(positions[joint_index]),
                    ],
                    fill=color,
                    width=4,
                )
            if not compact:
                drawing.line(
                    [(offset + 15, 515), (offset + 465, 515)], fill="#bdc9d3", width=1
                )
                drawing.text(
                    (offset + 18, 528),
                    "World XY: root / left ankle / right ankle",
                    font=small,
                    fill="#536477",
                )
                drawing.text(
                    (offset + 18, 780),
                    f"Frame {actual_frame + 1}/{len(sequence)} | 20 fps",
                    font=small,
                    fill="#536477",
                )
                for body_index, color in enumerate(colors):
                    points = (
                        paths[column][: actual_frame + 1, body_index] - path_center
                    ) * [path_scale, -path_scale] + [offset + 240, 655]
                    if len(points) > 1:
                        drawing.line(
                            [tuple(point) for point in points], fill=color, width=2
                        )
                    endpoint = points[-1]
                    drawing.ellipse(
                        [tuple(endpoint - 3), tuple(endpoint + 3)], fill=color
                    )
                drawing.text(
                    (offset + 365, 748), "X right, Y up", font=small, fill="#536477"
                )
        return canvas

    video = output / f"{name}.mp4"
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
            "1440x820",
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
            "veryfast",
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
    for frame_index in range(frames):
        encoder.stdin.write(draw_frame(frame_index).tobytes())
    encoder.stdin.close()
    if encoder.wait():
        raise RuntimeError(f"Encoding failed: {name}")
    sample_frames = np.linspace(0, frames - 1, 6).astype(int)
    contact = Image.new("RGB", (1440, 2700), "white")
    for row_index, frame_index in enumerate(sample_frames):
        contact.paste(draw_frame(int(frame_index), compact=True), (0, row_index * 450))
    contact.save(output / f"{name}_contact.jpg", quality=92)
    probe = json.loads(
        subprocess.check_output(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_frames,r_frame_rate,width,height",
                "-of",
                "json",
                str(video),
            ]
        )
    )["streams"][0]
    assert int(probe["nb_frames"]) == frames and probe["r_frame_rate"] == "20/1"
    return dict(
        name=name,
        file=video.name,
        sha256=digest(video),
        frames=frames,
        fps=20,
        selected_commands=commands,
        selected_measured=measured,
        unit=unit,
        contact_frames=sample_frames.tolist(),
        source_lengths=[len(sequence) for sequence in sequences],
    )


@hydra.main(
    version_base="1.3", config_path="../config", config_name="render_stage2_review"
)
def main(config):
    sys.path.insert(0, str(config.code_root))
    from shared_motion.training.catalog import TASK_NAMES, measure
    from shared_motion.training.geometry import Skeleton
    from audit_sliding import measure_sliding

    torch.set_num_threads(4)
    root = Path(config.run)
    output = Path(config.output)
    output.mkdir(parents=True, exist_ok=True)
    if (output / "render_manifest.json").exists():
        raise FileExistsError(
            "Completed render already exists; use another output directory"
        )
    best = json.loads((root / "reports/best.json").read_text())
    assert best["step"] == config.step
    assert best["sha256"] == config.checkpoint_sha256 == digest(best["checkpoint"])
    scan = json.loads((root / f"reports/scans/scan_{config.step:06d}.json").read_text())
    assert digest(scan["motion_archive"]) == scan["motion_sha256"]
    archive = np.load(scan["motion_archive"])
    training = OmegaConf.load(root / "reports/config.yaml")
    skeleton = Skeleton(training.data.skeleton)
    parents = skeleton.parents + [20, 21]
    motion = torch.from_numpy(archive["motion"])
    lengths = torch.from_numpy(archive["lengths"])
    tasks = torch.from_numpy(archive["task"])
    with torch.no_grad():
        remeasured = measure(skeleton, motion, tasks, lengths).numpy()
    error = float(np.abs(remeasured - archive["measured"]).max())
    assert error < 1e-4, error
    validation = json.loads(Path(training.data.val_manifest).read_text())
    if isinstance(validation, dict):
        validation = validation["records"]
    captions = {row["key"]: row.get("caption", "") for row in validation}
    manifest = dict(
        best=best,
        source_scan=scan["motion_archive"],
        source_sha256=scan["motion_sha256"],
        skeleton_sha256=digest(training.data.skeleton),
        frozen_code=str(config.code_root),
        maximum_recomputed_quantity_error=error,
        fps=20,
        selection_rule="Non-turn indices0,4,9; left-turn indices4,2,0; right-turn5,7,9. No seed or sample quality selection.",
        description="Unmodified saved pure-noise DDIM scan, same text/noise per task or turn direction; raw human skeleton, not G1 simulation.",
        tasks=[],
        videos=[],
    )
    units = {
        name: (
            "m/s"
            if name in ["strike", "walk", "back_walk", "jog"]
            else "rad" if name in ["turn", "lean", "bow", "twist"] else "m"
        )
        for name in TASK_NAMES
    }
    for entry in scan["tasks"]:
        task_name = entry["task"]
        indices = np.flatnonzero(archive["task"] == TASK_NAMES.index(task_name))
        assert len(indices) == 10
        sequences = []
        for sample_index in indices:
            with torch.no_grad():
                sequence = skeleton(
                    motion[sample_index : sample_index + 1, : lengths[sample_index]]
                )[0].numpy()
            assert np.isfinite(sequence).all()
            sequences.append(sequence)
        requested = archive["requested"][indices]
        measured = remeasured[indices]
        sliding = [measure_sliding(sequence) for sequence in sequences]
        record = dict(
            entry,
            unit=units[task_name],
            endpoint_gain=float(
                (measured[-1] - measured[0]) / (requested[-1] - requested[0])
            ),
            increasing_intervals=int((np.diff(measured) > 0).sum()),
            sliding=sliding,
            captions=[captions.get(key, "") for key in entry["keys"]],
        )
        manifest["tasks"].append(record)
        groups = (
            [(task_name, [0, 4, 9])]
            if task_name != "turn"
            else [("turn_left", [4, 2, 0]), ("turn_right", [5, 7, 9])]
        )
        for name, selected in groups:
            video = render_group(
                output,
                name,
                [sequences[index] for index in selected],
                parents,
                requested[selected].tolist(),
                measured[selected].tolist(),
                units[task_name],
                config.step,
            )
            video.update(
                task=task_name,
                scan_rows=indices[selected].tolist(),
                point_indices=selected,
                caption=record["captions"][selected[0]],
                noise_seed=entry["noise_seed"],
            )
            manifest["videos"].append(video)
            print(
                json.dumps(
                    dict(
                        rendered=name,
                        requested=video["selected_commands"],
                        measured=video["selected_measured"],
                    )
                ),
                flush=True,
            )
    assert len(manifest["tasks"]) == 20 and len(manifest["videos"]) == 21
    (output / "render_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2)
    )
    cards = []
    for video in manifest["videos"]:
        task = next(
            entry for entry in manifest["tasks"] if entry["task"] == video["task"]
        )
        levels = " · ".join(
            f'{label} 请求 {requested:.3f} → 实测 {measured:.3f} {video["unit"]}'
            for label, requested, measured in zip(
                ["低", "中", "高"],
                video["selected_commands"],
                video["selected_measured"],
            )
        )
        cards.append(
            f'<section id="{video["name"]}"><h2>{video["name"]}</h2><p>{levels}</p><video controls loop playsinline preload="metadata" src="{video["file"]}"></video><p>十档 MAE {task["mae"]:.4f} {task["unit"]}；端点响应增益 {task["endpoint_gain"]:.2f}（理想约 1）</p><details><summary>文本与来源</summary><p>{html.escape(video["caption"])}</p><p>seed {video["noise_seed"]}，scan rows {video["scan_rows"]}</p><a href="{video["name"]}_contact.jpg">六时刻抽帧</a></details></section>'
        )
    navigation = " ".join(
        f'<a href="#{video["name"]}">{video["name"]}</a>'
        for video in manifest["videos"]
    )
    (output / "index.html").write_text(
        '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>20任务三档复核 · Stage2 58000</title><style>body{font:16px system-ui;max-width:1440px;margin:32px auto;padding:0 20px;background:#edf2f7;color:#17324a}section{background:white;padding:24px;margin:24px 0;border-radius:14px}video{width:100%}nav{line-height:2.2}nav a{margin-right:14px}p{line-height:1.7}</style><h1>20 任务 · 低／中／高三档复核</h1><p>完整20任务训练，仅五类替换清洗数据。Stage2 58,000步最佳检查点；同一验证文本与噪声，只改变命令。20类、21组视频、63条动作；转身左右分组，侧步使用0.4–1.0m新范围。</p><p>原始人体骨架与世界坐标轨迹，20fps；没有重定位、稳定、落地或姿态修正。此页不是G1仿真。三档选固定扫描点0/4/9，中档沿用过去的离散点而非精确区间中点。固定相机与比例在每组三档间一致。</p><p><a href="render_manifest.json">检查点、动作来源、完整十档指标与哈希</a> · <a href="review.txt">检查结论</a></p><nav>'
        + navigation
        + "</nav>"
        + "".join(cards)
        + "</html>"
    )
    print("Complete:20 tasks,21 videos,63 displayed motions", flush=True)


if __name__ == "__main__":
    main()
