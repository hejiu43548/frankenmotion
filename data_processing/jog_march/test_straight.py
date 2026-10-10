"""Reject curved running even when it is forward relative to the body."""

from pathlib import Path
import unittest

import numpy as np
from omegaconf import OmegaConf

from data_processing.jog_march.straight import select_straight_windows
from data_processing.jog_march.straight_rules import straight_metrics
from data_processing.jog_march.test_rules import gait


class StraightRunningTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        directory = Path(__file__).resolve().parents[2] / "config"
        cls.path_rules = OmegaConf.load(directory / "jog_straight_repair.yaml").straight
        cls.gait_rules = OmegaConf.load(directory / "jog_march_repair.yaml").rules

    def test_straight_running_preserved(self):
        joints = gait(1.2)
        metrics, failures = straight_metrics(joints, self.path_rules)
        self.assertEqual(failures, [])
        windows, rejected = select_straight_windows(
            joints, self.gait_rules, self.path_rules
        )
        self.assertEqual(windows, [(0, len(joints))])

    def test_curved_path_rejected_with_body_aligned_to_curve(self):
        joints = gait(1.2)
        angles = np.linspace(0, np.pi / 2, len(joints))
        root = joints[:, 0, :2].copy()
        relative = joints[:, :, :2] - root[:, None]
        rotations = np.stack(
            [
                np.stack([np.cos(angles), -np.sin(angles)], axis=-1),
                np.stack([np.sin(angles), np.cos(angles)], axis=-1),
            ],
            axis=1,
        )
        trajectory = np.stack([4 * np.sin(angles), 4 * (1 - np.cos(angles))], axis=-1)
        joints[:, :, :2] = (
            np.einsum("tij,tkj->tki", rotations, relative) + trajectory[:, None]
        )
        metrics, failures = straight_metrics(joints, self.path_rules)
        self.assertIn("body_heading_turn_inside_clip", failures)
        self.assertIn("travel_direction_turn_inside_clip", failures)

    def test_s_bend_rejected_despite_matching_endpoint_heading(self):
        joints = gait(1.2)
        joints[:, :, 1] += (0.35 * np.sin(np.linspace(0, 2 * np.pi, len(joints))))[
            :, None
        ]
        metrics, failures = straight_metrics(joints, self.path_rules)
        self.assertIn("deviates_from_start_end_line", failures)

    def test_world_rotation_and_translation_do_not_change_decision(self):
        joints = gait(1.2)
        rotated = joints.copy()
        angle = 1.1
        rotation = np.array(
            [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]]
        )
        rotated[:, :, :2] = joints[:, :, :2] @ rotation.T + np.array([7, -3])
        original, failures = straight_metrics(joints, self.path_rules)
        transformed, failures = straight_metrics(rotated, self.path_rules)
        self.assertEqual(failures, [])
        for name in original:
            self.assertAlmostEqual(original[name], transformed[name], places=3)

    def test_straight_path_with_body_rotation_rejected(self):
        joints = gait(1.2)
        angles = np.linspace(0, np.pi / 3, len(joints))
        joints[:, 1, :2] = joints[:, 0, :2] + np.stack(
            [-0.1 * np.sin(angles), 0.1 * np.cos(angles)], axis=-1
        )
        joints[:, 2, :2] = 2 * joints[:, 0, :2] - joints[:, 1, :2]
        metrics, failures = straight_metrics(joints, self.path_rules)
        self.assertIn("body_heading_turn_inside_clip", failures)


if __name__ == "__main__":
    unittest.main()
