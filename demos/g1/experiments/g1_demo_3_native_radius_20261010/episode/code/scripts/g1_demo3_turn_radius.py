"""Fit circles to actual turn paths; report residuals and chord measures too."""

import json
from pathlib import Path
import sys

import numpy as np


def measure(folder):
    folder = Path(folder)
    states = np.load(folder / "actual.npz")["qpos"]
    composition = json.loads((folder / "composition.json").read_text())
    records = []
    for segment in composition["segments"]:
        if not segment["name"].startswith("turn"):
            continue
        first, last = [round(segment[name] * 50) for name in ["start", "end"]]
        points = states[first : last + 1, :2]
        system = np.c_[2 * points, np.ones(len(points))]
        solution = np.linalg.lstsq(system, np.sum(points**2, axis=1), rcond=None)[0]
        radius = float(np.sqrt(solution[2] + np.sum(solution[:2] ** 2)))
        residual = float(
            np.sqrt(
                np.mean((np.linalg.norm(points - solution[:2], axis=1) - radius) ** 2)
            )
        )
        records.append(
            dict(
                name=segment["name"],
                fitted_radius_m=radius,
                radial_rmse_m=residual,
                half_endpoint_chord_m=float(np.linalg.norm(points[-1] - points[0]) / 2),
                actual_path_m=float(
                    np.linalg.norm(np.diff(points, axis=0), axis=1).sum()
                ),
            )
        )
    return records


def compare(folder, baseline):
    original = measure(baseline)
    candidate = measure(folder)
    report = dict(
        baseline=original,
        candidate=candidate,
        radius_ratios=[
            current["fitted_radius_m"] / previous["fitted_radius_m"]
            for previous, current in zip(original, candidate)
        ],
        definition="Least-squares circle fit over each actual pelvis XY turn path. Motions are not perfect semicircles; radial residual and half-chord are reported separately.",
    )
    (Path(folder) / "radius_comparison.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    compare(sys.argv[1], sys.argv[2])
