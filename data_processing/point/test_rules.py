"""Regression checks for semantic false positives and retraction contamination."""

from pathlib import Path
import unittest

import numpy as np
from omegaconf import OmegaConf

from data_processing.point.repair import compatible_event, crop_metrics


class ReachAdmissionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.rules = OmegaConf.load(
            Path(__file__).resolve().parents[2] / "config/reach_xyz_repair.yaml"
        ).rules

    def test_starting_point_is_not_pointing(self):
        self.assertFalse(
            compatible_event(
                dict(
                    label=dict(
                        proc_label="walk back to starting point", act_cat=["walk"]
                    )
                )
            )
        )
        self.assertFalse(
            compatible_event(
                dict(label=dict(proc_label="point with left hand", act_cat=["point"]))
            )
        )
        self.assertTrue(
            compatible_event(
                dict(label=dict(proc_label="point with right hand", act_cat=["point"]))
            )
        )
        self.assertFalse(
            compatible_event(dict(label=dict(proc_label="point forward", act_cat=[])))
        )

    def posture(self):
        positions = np.zeros((30, 24, 3), dtype=np.float32)
        positions[:, 0, 2] = 0.9
        positions[:, 1, 1] = 0.1
        positions[:, 2, 1] = -0.1
        positions[:, [16, 17], 2] = 1.3
        positions[:, 17, 1] = -0.18
        positions[:, 19] = [0.25, -0.18, 1.1]
        positions[:, 20] = [0, 0.18, 0.75]
        positions[:, 21, 0] = np.minimum(np.linspace(0, 0.8, 30), 0.5)
        positions[:, 21, 1] = -0.18
        positions[:, 21, 2] = 0.9
        return positions

    def test_retraction_from_raised_hand_is_rejected(self):
        positions = self.posture()
        positions[:3, 21, 2] = 1.5
        metrics, failures = crop_metrics(positions, self.rules)
        self.assertIn("starts_from_existing_target_or_retraction", failures)

    def test_static_raised_left_arm_is_rejected(self):
        positions = self.posture()
        positions[:, 20, 2] = 1.5
        metrics, failures = crop_metrics(positions, self.rules)
        self.assertIn("raised_noncontrolled_arm", failures)

    def test_relaxed_hand_is_not_a_target(self):
        positions = self.posture()
        positions[-5:, 21] = [0, -0.2, 0.75]
        metrics, failures = crop_metrics(positions, self.rules)
        self.assertIn("target_is_relaxed_hand_position", failures)


if __name__ == "__main__":
    unittest.main()
