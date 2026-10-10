"""Verify that main action segments preserve native GMR trajectories."""

import json
from pathlib import Path
import sys

import numpy as np
from scipy.spatial.transform import Rotation

from g1_demo3_phase import phase_resample
from g1_demo_physics import place


def audit(folder, source_root):
    folder = Path(folder)
    composition = json.loads((folder / "composition.json").read_text())
    reference = np.load(folder / "reference.npz")["qpos"]
    records = []
    for segment, transform in zip(
        composition["segments"], composition["transformations"]
    ):
        original = np.load(
            Path(source_root) / "retarget" / (transform["source"] + ".npz")
        )["qpos"]
        first, last = transform["source_crop_frames"]
        expected = phase_resample(original[first : last + 1], 1.0)
        actual = reference[
            round(segment["start"] * 50) : round(segment["end"] * 50) + 1
        ]
        assert len(expected) == len(actual)
        yaw_delta = (
            Rotation.from_quat(actual[0, [4, 5, 6, 3]]).as_euler("xyz")[2]
            - Rotation.from_quat(expected[0, [4, 5, 6, 3]]).as_euler("xyz")[2]
        )
        expected = place(expected, yaw_delta, [0, 0, 0])
        expected[:, :3] += actual[0, :3] - expected[0, :3]
        orientation_errors = (
            Rotation.from_quat(expected[:, [4, 5, 6, 3]]).inv()
            * Rotation.from_quat(actual[:, [4, 5, 6, 3]])
        ).magnitude()
        record = dict(
            name=segment["name"],
            max_joint_error_rad=float(np.abs(expected[:, 7:] - actual[:, 7:]).max()),
            max_root_error_m=float(np.abs(expected[:, :3] - actual[:, :3]).max()),
            max_orientation_error_rad=float(orientation_errors.max()),
        )
        assert (
            max(record[key] for key in record if key.startswith("max_")) < 1e-9
        ), record
        assert transform["heading_correction_deg"] == 0 and transform["speed"] == 1
        records.append(record)
    report = dict(
        passed=True,
        scope="Each main segment equals one contiguous native GMR crop at original timing, with only constant world yaw/translation and 20-to-50Hz interpolation. Segment-boundary bridges and terminal hold are separate.",
        segments=records,
    )
    (folder / "native_preservation.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report))


if __name__ == "__main__":
    audit(sys.argv[1], sys.argv[2])
