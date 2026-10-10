"""Synthetic gait regression fixtures; never included in training manifests."""

from pathlib import Path
import hashlib
import json
import tempfile
import unittest

import numpy as np
from omegaconf import OmegaConf
import torch

from data_processing.jog_march.rules import classify_clip
from data_processing.jog_march.rules import semantic_failures
from data_processing.jog_march.rules import trim_to_running_cycles
from shared_motion.training.catalog import new_quantity
from scripts.prepare_amass20 import load_running_records


def gait(speed=0.0, stance_fraction=0.35, synchronized=False):
    times = np.arange(100) / 20
    frequency = 1.6
    joints = np.zeros((len(times), 24, 3), dtype=np.float32)
    joints[:, :, 0] = (times * speed)[:, None]
    joints[:, 0, 2] = 0.9
    joints[:, [1, 2], 2] = 0.85
    joints[:, [16, 17], 2] = 1.4
    joints[:, [1, 7, 10, 16], 1] = 0.10
    joints[:, [2, 8, 11, 17], 1] = -0.10
    for foot, offset in [(0, 0.0), (1, 0.0 if synchronized else 0.5)]:
        cycles = times * frequency + offset
        phase = cycles % 1
        swing = np.clip((phase - stance_fraction) / (1 - stance_fraction), 0, 1)
        height = 0.20 * np.sin(np.pi * swing)
        anchor = (np.floor(cycles) - offset) / frequency * speed
        horizontal = anchor + (speed / frequency) * (3 * swing**2 - 2 * swing**3)
        joints[:, [7 + foot, 10 + foot], 0] = horizontal[:, None]
        joints[:, 7 + foot, 2] = height + 0.04
        joints[:, 10 + foot, 2] = height
    return joints


class RunningClassificationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = OmegaConf.load(
            Path(__file__).resolve().parents[2] / "config/jog_march_repair.yaml"
        ).rules

    def test_forward_running_is_jog(self):
        task, metrics, failures = classify_clip(gait(1.2), self.rules)
        self.assertEqual(failures, [])
        self.assertEqual(task, "jog")

    def test_in_place_running_is_march(self):
        task, metrics, failures = classify_clip(gait(), self.rules)
        self.assertEqual(failures, [])
        self.assertEqual(task, "march")

    def test_walking_is_not_running(self):
        task, metrics, failures = classify_clip(
            gait(0.9, stance_fraction=0.65), self.rules
        )
        self.assertIsNone(task)
        self.assertIn("no_repeated_running_flight_evidence", failures)

    def test_standing_is_not_stationary_running(self):
        joints = gait()
        joints[:] = joints[0]
        task, metrics, failures = classify_clip(joints, self.rules)
        self.assertIsNone(task)
        self.assertIn("insufficient_bilateral_stride_cycles", failures)

    def test_standing_prefix_trimmed_not_relabelled(self):
        running = gait()
        prefix = np.repeat(running[:1], 30, axis=0)
        prefix[:, [7, 8], 2] = 0.04
        prefix[:, [10, 11], 2] = 0
        mixed = np.concatenate([prefix, running])
        bounds = trim_to_running_cycles(mixed, self.rules)
        self.assertIsNotNone(bounds)
        self.assertGreater(bounds[0], 20)

    def test_bilateral_hopping_rejected(self):
        task, metrics, failures = classify_clip(gait(synchronized=True), self.rules)
        self.assertIsNone(task)
        self.assertIn("synchronous_hopping_instead_of_running", failures)

    def test_treadmill_sliding_rejected(self):
        joints = gait(1.2)
        joints[:, :, 0] -= joints[:, 0:1, 0]
        task, metrics, failures = classify_clip(joints, self.rules)
        self.assertIsNone(task)
        self.assertIn("support_foot_sliding_or_treadmill_motion", failures)

    def test_backward_running_rejected(self):
        task, metrics, failures = classify_clip(gait(-1.2), self.rules)
        self.assertIsNone(task)
        self.assertIn("not_continuous_forward_running", failures)

    def test_slow_travel_not_mislabelled_stationary(self):
        task, metrics, failures = classify_clip(gait(0.2), self.rules)
        self.assertIsNone(task)
        self.assertIn("ambiguous_travel_or_stationarity", failures)

    def test_categories_and_bad_descriptions(self):
        self.assertEqual(
            semantic_failures(dict(act_cat=["jog"], proc_label="jog in place")), []
        )
        for categories, label in [
            ([], "running"),
            (["walk"], "walk"),
            (["run"], "pretend to run"),
            (["run"], "run on treadmill"),
            (["run"], "run backwards"),
        ]:
            self.assertTrue(
                semantic_failures(dict(act_cat=categories, proc_label=label))
            )

    def test_march_height_independent_of_crop_phase_and_vertical_origin(self):
        joints = torch.from_numpy(gait())[None]
        expected = new_quantity(joints, "march")
        shifted = torch.roll(joints, 13, dims=1)
        shifted[:, :, :, 2] += 1.5
        torch.testing.assert_close(new_quantity(shifted, "march"), expected)
        self.assertGreater(float(expected[0]), 0.15)

    def test_legacy_regex_manifests_cannot_be_reimported(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "train.json"
            path.write_text(
                json.dumps(
                    [dict(task=task, split="train") for task in ["jog", "march"]]
                )
            )
            with self.assertRaisesRegex(ValueError, "Legacy regex-only"):
                load_running_records(directory, "train")

    def test_repaired_import_requires_exact_cache_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            cache = root / "fixture.npz"
            cache.write_bytes(b"checksum fixture, not training data")
            records = [
                dict(
                    task=task,
                    split="train",
                    semantic_revision=revision,
                    cache=str(cache),
                    cache_sha256=hashlib.sha256(cache.read_bytes()).hexdigest(),
                )
                for task, revision in [
                    ("jog", "straight_running_v2"),
                    ("march", "stationary_running_v1"),
                ]
            ]
            (root / "train.json").write_text(json.dumps(records))
            self.assertEqual(load_running_records(root, "train"), records)
            cache.write_bytes(b"tampered fixture")
            with self.assertRaisesRegex(ValueError, "Invalid repaired running cache"):
                load_running_records(root, "train")


if __name__ == "__main__":
    unittest.main()
