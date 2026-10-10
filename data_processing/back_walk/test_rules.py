"""Check direction in body coordinates and temporal crop boundaries."""

from pathlib import Path
import unittest

import numpy as np
from omegaconf import OmegaConf

from data_processing.back_walk.repair import backward_windows
from data_processing.back_walk.repair import check_clip


class BackwardCropTests(unittest.TestCase):
    def setUp(self):
        self.rules = OmegaConf.load(
            Path(__file__).parents[2] / "config/back_walk_repair.yaml"
        ).rules

    def motion(self, velocity):
        joints = np.zeros((len(velocity) + 1, 24, 3), np.float32)
        joints[:, :, 0] = np.concatenate([[0], np.cumsum(velocity) / 20])[:, None]
        joints[:, 16, 2] = 0.45
        joints[:, 17, 2] = 0.45
        joints[:, 1, 1] = 0.1
        joints[:, 2, 1] = -0.1
        joints[:, 7, 2] = 0.06 * np.sin(np.arange(len(joints)) * 0.3)
        joints[:, 8, 2] = 0.06 * np.cos(np.arange(len(joints)) * 0.3)
        return joints

    def test_bending_backward_motion_is_rejected(self):
        joints = self.motion(np.full(100, -0.5))
        joints[:, [16, 17], 0] += 0.65
        self.assertIn(
            "bending_crouching_instead_of_upright_walking",
            check_clip(joints, self.rules)[1],
        )

    def test_forward_has_no_backward_windows(self):
        self.assertEqual(
            backward_windows(self.motion(np.full(100, 0.5)), self.rules), []
        )

    def test_mixed_motion_is_cropped(self):
        joints = self.motion(
            np.concatenate([np.full(50, 0.5), np.full(100, -0.5), np.full(50, 0.5)])
        )
        windows = backward_windows(joints, self.rules)
        self.assertTrue(windows)
        for start, stop in windows:
            self.assertGreaterEqual(start, 50)
            self.assertLessEqual(stop, 151)
            self.assertEqual(check_clip(joints[start:stop], self.rules)[1], [])

    def test_world_negative_forward_walk_is_not_backward(self):
        joints = self.motion(np.full(100, -0.5))
        joints[:, 1, 1] = -0.1
        joints[:, 2, 1] = 0.1
        self.assertEqual(backward_windows(joints, self.rules), [])

    def test_stationary_and_short_reverse_are_rejected(self):
        self.assertEqual(backward_windows(self.motion(np.zeros(100)), self.rules), [])
        self.assertEqual(
            backward_windows(self.motion(np.full(20, -0.5)), self.rules), []
        )


if __name__ == "__main__":
    unittest.main()
