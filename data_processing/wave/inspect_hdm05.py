"""Read native05-01 sources; save actual FK and temporal diagnostics."""

import json
import sys
from pathlib import Path
import numpy as np
import torch
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.prepare_amass20 import convert_source
from shared_motion.training.geometry import Skeleton

root = Path("/home/psirobot/projects/frankenmotion")
out = Path("/mnt/sda2/frankenmotion/outputs_amass/hdm05_wave_boundary_audit_20261009")
out.mkdir(exist_ok=True)
skpath = (
    root
    / "outputs_amass/transfer_charlie_20261008/snapshot/outputs_amass/franken_eleven_20261003/skeleton.npz"
)
sk = Skeleton(skpath)
torch.set_num_threads(2)
annroot = root / "datasets/annotations/frankenstein-dataset/annotations"
anns = json.loads((annroot / "annotations.json").read_text())
splits = {
    anns[k]["path"]: s
    for s in ["train", "val", "test"]
    for k in (annroot / "splits" / f"{s}.txt").read_text().split()
}
rows = []
for raw in sorted(Path("/mnt/sda2/dataset/amass/MPI_HDM05").glob("*/*05-01*.npz")):
    family = str(raw.relative_to("/mnt/sda2/dataset/amass").with_suffix(""))
    split = splits.get(family)
    if split != "train":
        rows.append(dict(family=family, split=split, status="held_out_no_inspection"))
        continue
    full = (
        root / "outputs_amass/unified_direct_20261008/data/motions" / (family + ".npy")
    )
    if not full.exists():
        full = out / "motions" / (family + ".npy")
        result = convert_source(
            (
                str(root / "work/unified_direct/prepare_data.py"),
                (family, str(raw), str(full), str(skpath)),
            )
        )
        assert result["status"] in ["ok", "cached"], result
    motion = np.load(full)
    with torch.no_grad():
        j = sk(torch.from_numpy(motion)[None])[0].numpy()
    np.savez(
        out / (raw.stem + ".npz"), joints=j, parents=np.array(sk.parents + [20, 21])
    )
    times = np.arange(len(j)) / 20
    fig, axes = plt.subplots(3, 1, figsize=(15, 7), sharex=True, layout="constrained")
    for wrist, color, name in [(20, "#d54b42", "left"), (21, "#2375ba", "right")]:
        rel = j[:, wrist] - j[:, 0]
        axes[0].plot(times, rel[:, 1], color=color, label=name + " lateral")
        axes[1].plot(times, rel[:, 2], color=color, label=name + " height above root")
    for foot, color, name in [
        (7, "#d54b42", "left ankle"),
        (8, "#2375ba", "right ankle"),
        (0, "#555555", "root"),
    ]:
        axes[2].plot(
            times,
            np.linalg.norm(j[:, foot, :2] - j[:1, foot, :2], axis=-1),
            color=color,
            label=name + " displacement",
        )
    for ax in axes:
        ax.legend()
        ax.grid(alpha=0.2)
    axes[-1].set_xlabel("Source time (s)")
    axes[-1].set_xticks(np.arange(0, times[-1] + 1, 2))
    fig.suptitle(raw.stem + " | unchanged FK | training split only")
    fig.savefig(out / (raw.stem + ".png"), dpi=100)
    plt.close(fig)
    rows.append(
        dict(
            family=family,
            split=split,
            motion_source=str(full),
            frames=len(motion),
            seconds=len(motion) / 20,
            status="diagnostic",
        )
    )
(out / "inventory.json").write_text(json.dumps(rows, indent=2))
print(json.dumps(rows, indent=2))
