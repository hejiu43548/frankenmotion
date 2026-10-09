"""Audit completed single-task runs and evaluate fixed-context command responses."""

import html
import json
from pathlib import Path
import subprocess
import sys

import hydra
import matplotlib
import numpy as np
from omegaconf import OmegaConf
from PIL import Image
from PIL import ImageDraw
from PIL import ImageFont
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def response_metrics(requested, measured):
    return dict(
        mae=float(np.abs(measured - requested).mean()),
        endpoint_gain=float(
            (measured[-1] - measured[0]) / (requested[-1] - requested[0])
        ),
        increasing_intervals=int((np.diff(measured) > 0).sum()),
        requested=requested.tolist(),
        measured=measured.tolist(),
    )


def render_video(output, task_name, joints, parents, requested, measured, step):
    selected_points = [0, 4, 9]
    projection = np.array([[0.707, -0.707, 0], [0.354, 0.354, -0.866]])
    selected = joints[selected_points]
    projected = selected @ projection.T
    center = (projected.min((0, 1, 2)) + projected.max((0, 1, 2))) / 2
    scale = min(230, 390 / max(float(np.ptp(projected, axis=(0, 1, 2)).max()), 0.1))
    font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 19)
    small = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", 15)
    left_joints = {1, 4, 7, 10, 13, 16, 18, 20, 22}
    right_joints = {2, 5, 8, 11, 14, 17, 19, 21, 23}

    def frame_image(frame_index):
        canvas = Image.new("RGB", (1440, 620), "#f5f8fb")
        drawing = ImageDraw.Draw(canvas)
        drawing.text(
            (18, 12),
            f"{task_name} | stage2 final step {step} | t={frame_index/20:.2f}s",
            font=font,
            fill="#17324a",
        )
        drawing.text(
            (18, 42),
            "Same held-out text and noise; only command changes. Raw FK at 20fps. Red: left / blue: right.",
            font=small,
            fill="#536477",
        )
        for column, point_index in enumerate(selected_points):
            drawing.text(
                (column * 480 + 18, 78),
                f"Request {requested[point_index]:.3f} | actual {measured[point_index]:.3f}",
                font=font,
                fill="#17324a",
            )
            positions = (projected[column, frame_index] - center) * scale + np.array(
                [column * 480 + 240, 355]
            )
            for joint_index in range(1, 24):
                color = (
                    "#d54b42"
                    if joint_index in left_joints
                    else "#2375ba" if joint_index in right_joints else "#354554"
                )
                drawing.line(
                    [
                        tuple(positions[parents[joint_index]]),
                        tuple(positions[joint_index]),
                    ],
                    fill=color,
                    width=5,
                )
            drawing.text(
                (column * 480 + 18, 583),
                f"Frame {frame_index+1}/{joints.shape[1]} | fixed common camera",
                font=small,
                fill="#536477",
            )
        return canvas

    video = output / f"{task_name}.mp4"
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
            "1440x620",
            "-r",
            "20",
            "-i",
            "-",
            "-an",
            "-c:v",
            "libx264",
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
    for frame_index in range(joints.shape[1]):
        encoder.stdin.write(frame_image(frame_index).tobytes())
    encoder.stdin.close()
    if encoder.wait():
        raise RuntimeError("Video encoding failed")
    contact = Image.new("RGB", (1440, 1860), "white")
    for row_index, frame_index in enumerate(
        np.linspace(0, joints.shape[1] - 1, 3).astype(int)
    ):
        contact.paste(frame_image(int(frame_index)), (0, row_index * 620))
    contact.save(output / f"{task_name}_contact.png")
    video_info = json.loads(
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
    assert int(video_info["nb_frames"]) == joints.shape[1]
    assert video_info["r_frame_rate"] == "20/1"
    return dict(file=video.name, frames=joints.shape[1], fps=20)


@hydra.main(
    version_base="1.3", config_path="../config", config_name="single_task_stage2"
)
def main(config):
    sys.path.insert(0, str(config.code_root))
    from shared_motion.training.catalog import COMMAND_RANGES
    from shared_motion.training.catalog import TASK_NAMES
    from shared_motion.training.catalog import measure
    from shared_motion.training.data import MotionDataset
    from shared_motion.training.geometry import Skeleton
    from shared_motion.training.model import build_model
    from shared_motion.training.model import file_sha256
    from shared_motion.training.runner import checkpoint_compatible
    from shared_motion.training.runner import save_json

    root = Path(config.output)
    output = root / "review"
    output.mkdir(exist_ok=True)
    torch.set_num_threads(4)
    torch.backends.mha.set_fastpath_enabled(False)
    summaries = []
    for task_name in config.tasks:
        report = root / task_name / "reports"
        status = json.loads((report / "status.json").read_text())
        assert status["state"] == "complete" and status["stage"] == 2
        run_config = OmegaConf.load(report / "config.yaml")
        checkpoint_path = (
            root / task_name / "artifacts" / f'checkpoint_{status["step"]:06d}.pt'
        )
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        model = build_model(run_config, "task").to("cuda").eval()
        checkpoint_compatible(checkpoint, model, [task_name], stage=2)
        model.load_adapter(checkpoint["adapter"])
        initial = torch.load(
            config.initial_root, map_location="cpu", weights_only=False
        )
        assert all(
            torch.equal(checkpoint["adapter"]["root." + name], value)
            for name, value in initial["adapter"].items()
        )
        optimizer_steps = {
            int(value["step"])
            for value in checkpoint["optimizer"]["state"].values()
            if "step" in value
        }
        assert optimizer_steps == {int(status["step"])}
        skeleton = Skeleton(config.skeleton).to("cuda")
        validation = MotionDataset(
            run_config.data.val_manifest, "val", [task_name], config.cache_root
        )
        task_index = TASK_NAMES.index(task_name)
        commands = torch.linspace(*COMMAND_RANGES[task_index], 10, device="cuda")
        requested = commands.cpu().numpy()
        contexts = []
        archive_arrays = []
        archive_lengths = []
        with torch.no_grad():
            for row_index, record in enumerate(validation.rows):
                for seed in [85101, 85102, 85103]:
                    batch = validation.batch([row_index] * 10, "cuda")
                    generated = model.sample(batch, commands, skeleton, [seed] * 10, 50)
                    actual = measure(
                        skeleton, generated, batch["task"], batch["lengths"]
                    )
                    assert (
                        torch.isfinite(generated).all() and torch.isfinite(actual).all()
                    )
                    contexts.append(
                        dict(
                            key=record["key"],
                            family=record["family"],
                            seed=seed,
                            **response_metrics(requested, actual.cpu().numpy()),
                        )
                    )
                    archive_arrays.append(generated.cpu().numpy())
                    archive_lengths.append(batch["lengths"].cpu().numpy())
        motion_path = root / task_name / "artifacts" / "response_multiseed.npz"
        np.savez_compressed(
            motion_path,
            motion=np.stack(archive_arrays),
            lengths=np.stack(archive_lengths),
            requested=requested,
            measured=np.array([context["measured"] for context in contexts]),
        )
        with np.load(motion_path) as archive:
            maximum_error = 0.0
            with torch.no_grad():
                for context_index, context in enumerate(contexts):
                    generated = torch.tensor(
                        archive["motion"][context_index], device="cuda"
                    )
                    lengths = torch.tensor(
                        archive["lengths"][context_index], device="cuda"
                    )
                    recomputed = (
                        measure(
                            skeleton,
                            generated,
                            torch.full((10,), task_index, device="cuda"),
                            lengths,
                        )
                        .cpu()
                        .numpy()
                    )
                    maximum_error = max(
                        maximum_error,
                        float(
                            np.abs(
                                recomputed - archive["measured"][context_index]
                            ).max()
                        ),
                    )
        assert maximum_error < 1e-5
        scans = [
            json.loads(path.read_text())
            for path in sorted((report / "scans").glob("scan_*.json"))
        ]
        training_rows = json.loads((report / "data_index/train.json").read_text())
        training_values = np.array([record["quantity"] for record in training_rows])
        all_measured = np.array([context["measured"] for context in contexts])
        figure, axes = plt.subplots(1, 3, figsize=(16, 4.6), constrained_layout=True)
        for scan in scans:
            entry = scan["tasks"][0]
            if scan["step"] in [0, 1000, 5000, status["step"]]:
                axes[0].plot(
                    entry["requested"],
                    entry["measured"],
                    marker=".",
                    label=str(scan["step"]),
                )
        axes[0].plot(requested, requested, "k--", label="ideal")
        axes[0].set(
            title=f"{task_name}: fixed-context training scans",
            xlabel="Requested command",
            ylabel="Measured command",
        )
        axes[0].legend()
        for actual in all_measured:
            axes[1].plot(requested, actual, alpha=0.25, color="steelblue")
        axes[1].plot(requested, all_measured.mean(0), color="navy", label="mean")
        axes[1].plot(requested, requested, "k--", label="ideal")
        axes[1].set(
            title=f"Final: {len(validation.rows)} val texts x 3 new seeds",
            xlabel="Requested command",
            ylabel="Measured command",
        )
        axes[1].legend()
        boundaries = np.linspace(*COMMAND_RANGES[task_index], 11)
        axes[2].hist(training_values, bins=boundaries, color="teal", edgecolor="white")
        axes[2].set(
            title=f"Train support: {len(training_values)} total (outside omitted)",
            xlabel="Measured training command",
            ylabel="Records per bin",
        )
        for axis in axes:
            axis.grid(alpha=0.2)
        figure.savefig(output / f"{task_name}_response.png", dpi=150)
        plt.close(figure)
        final_scan = scans[-1]
        assert file_sha256(final_scan["motion_archive"]) == final_scan["motion_sha256"]
        with np.load(final_scan["motion_archive"]) as archive:
            lengths = archive["lengths"]
            assert len(set(lengths.tolist())) == 1
            with torch.no_grad():
                joints = (
                    skeleton(torch.tensor(archive["motion"], device="cuda"))
                    .cpu()
                    .numpy()[:, : int(lengths[0])]
                )
            video = render_video(
                output,
                task_name,
                joints,
                skeleton.parents + [20, 21],
                archive["requested"],
                archive["measured"],
                status["step"],
            )
        video["sha256"] = file_sha256(output / video["file"])
        summary = dict(
            task=task_name,
            step=status["step"],
            train=len(training_values),
            val=len(validation.rows),
            checkpoint=str(checkpoint_path),
            checkpoint_sha256=file_sha256(checkpoint_path),
            initial_scan=scans[0]["tasks"][0],
            final_scan=final_scan["tasks"][0],
            best=json.loads((report / "best.json").read_text()),
            contexts=contexts,
            multiseed=dict(
                mae=float(np.abs(all_measured - requested).mean()),
                endpoint_gain=float(
                    np.mean([context["endpoint_gain"] for context in contexts])
                ),
                increasing_fraction=float(np.mean(np.diff(all_measured, axis=1) > 0)),
                measured_mean=all_measured.mean(0).tolist(),
            ),
            video=video,
            verification=dict(
                root_parameters_exact=True,
                optimizer_steps=list(optimizer_steps),
                saved_motion_recompute_max_error=maximum_error,
            ),
            completion_audit=json.loads((report / "completion_audit.json").read_text()),
            motion_archive=str(motion_path),
            motion_sha256=file_sha256(motion_path),
        )
        save_json(output / f"{task_name}_assessment.json", summary)
        summaries.append(summary)
        print(
            json.dumps(dict(task=task_name, multiseed=summary["multiseed"])), flush=True
        )
        del model, checkpoint, initial
        torch.cuda.empty_cache()
    summaries = [
        json.loads(path.read_text())
        for path in sorted(output.glob("*_assessment.json"))
    ]
    save_json(
        output / "assessment.json",
        dict(
            tasks=summaries,
            limitations=[
                "Small cleaned datasets; no stage1 or stage3 training.",
                "Three new noise seeds over all held-out validation texts; not an untouched test set.",
                "Metrics do not certify semantic correctness or natural motion.",
                "Command intervals without training examples remain extrapolation/interpolation with sparse support.",
            ],
        ),
    )
    sections = "".join(
        f'<section><h2>{html.escape(summary["task"])}</h2><img src="{summary["task"]}_response.png"><video controls loop preload="metadata" src="{summary["task"]}.mp4"></video><p>Final 10,000 steps; raw 20fps skeleton. <a href="{summary["task"]}_assessment.json">Audit JSON</a></p></section>'
        for summary in summaries
    )
    (output / "index.html").write_text(
        '<!doctype html><meta charset="utf-8"><title>Single-task stage2 response</title><style>body{font:16px system-ui;max-width:1400px;margin:32px auto;background:#f3f6f9;color:#17324a}section{background:white;padding:22px;margin:20px 0}img,video{width:100%}</style><h1>Wave / strike / kick: independent stage2 training</h1><p>Frozen official backbone and imported root; independent TaskControl; 10,000 steps each, batch 8. Curves show 10 command values with identical text/noise within each curve. Validation texts are limited; command accuracy alone does not establish natural motion.</p>'
        + sections
    )


if __name__ == "__main__":
    main()
