"""Synthetic rule checks only; these fixtures are never training data."""

from pathlib import Path
import unittest

import numpy as np
from omegaconf import OmegaConf

from data_processing.jump.rules import check_clip
from data_processing.jump.rules import propose_windows
from data_processing.jump.rules import semantic_failures


def two_foot_jump():
    joints = np.zeros((36, 24, 3), dtype=np.float32)
    joints[:, 0, 2] = 0.9
    joints[:, [1, 2], 2] = 0.85
    joints[:, [4, 5], 2] = 0.43
    joints[:, [1, 4, 7, 10, 16], 1] = 0.1
    joints[:, [2, 5, 8, 11, 17], 1] = -0.1
    joints[:, [16, 17], 2] = 1.4
    height = np.zeros(36)
    height[10:23] = np.sin(np.linspace(0, np.pi, 13)) * 0.30
    joints[:, :, 2] += height[:, None]
    return joints


class JumpAdmissionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = OmegaConf.load(
            Path(__file__).resolve().parents[2] / "config/jump_repair.yaml"
        ).rules

    def test_complete_two_foot_jump(self):
        joints = two_foot_jump()
        self.assertEqual(check_clip(joints, self.rules)[1], [])
        self.assertTrue(propose_windows(joints, 0, len(joints), self.rules))

    def test_single_leg_rejected(self):
        joints = two_foot_jump()
        joints[:, [8, 11], 2] = 0
        self.assertIn(
            "not_exactly_one_bilateral_flight", check_clip(joints, self.rules)[1]
        )

    def test_different_landing_surface_rejected(self):
        joints = two_foot_jump()
        joints[23:, :, 2] += 0.20
        self.assertIn(
            "floor_height_difference_m_above_limit", check_clip(joints, self.rules)[1]
        )

    def test_staggered_takeoff_rejected(self):
        joints = two_foot_jump()
        joints[:, [8, 11], 2] = np.roll(joints[:, [8, 11], 2], 4, axis=0)
        self.assertIn(
            "asynchronous_takeoff_or_landing", check_clip(joints, self.rules)[1]
        )

    def test_jacks_spread_rejected(self):
        joints = two_foot_jump()
        joints[12:20, [7, 10], 1] += 0.3
        joints[12:20, [8, 11], 1] -= 0.3
        self.assertIn(
            "foot_separation_change_m_above_limit", check_clip(joints, self.rules)[1]
        )

    def test_category_required_and_variants_rejected(self):
        self.assertTrue(semantic_failures(dict(proc_label="jump", act_cat=[])))
        for label in [
            "jump rope",
            "jump off stairs",
            "hop on right foot",
            "jumping jacks",
            "jump left",
            "jump to the right",
            "hop backwards",
            "jump forward",
        ]:
            self.assertTrue(semantic_failures(dict(proc_label=label, act_cat=["jump"])))
        self.assertEqual(
            semantic_failures(dict(proc_label="jump in place", act_cat=["jump"])), []
        )

    def test_horizontal_travel_rejected(self):
        joints = two_foot_jump()
        joints[:, :, 0] += np.linspace(0, 0.5, len(joints))[:, None]
        self.assertIn(
            "root_horizontal_excursion_m_above_limit", check_clip(joints, self.rules)[1]
        )

    def test_jump_out_then_return_rejected(self):
        joints = two_foot_jump()
        offsets = np.zeros(len(joints))
        offsets[10:24] = np.linspace(0, 0.30, 14)
        offsets[24:] = np.linspace(0.30, 0, len(joints) - 24)
        joints[:, :, 1] += offsets[:, None]
        self.assertIn(
            "flight_landing_center_offset_m_above_limit",
            check_clip(joints, self.rules)[1],
        )

    def test_foot_shift_without_root_shift_rejected(self):
        joints = two_foot_jump()
        joints[23:, [7, 8, 10, 11], 1] += 0.2
        self.assertIn(
            "each_foot_landing_offset_m_above_limit", check_clip(joints, self.rules)[1]
        )

    def test_small_capture_drift_allowed(self):
        joints = two_foot_jump()
        joints[:, :, 0] += np.linspace(0, 0.025, len(joints))[:, None]
        self.assertEqual(check_clip(joints, self.rules)[1], [])

    def test_crouched_baseline_rejected(self):
        joints = two_foot_jump()
        joints[:3, [4, 5], 0] = 0.35
        self.assertIn("crop_starts_crouched", check_clip(joints, self.rules)[1])


if __name__ == "__main__":
    unittest.main()
