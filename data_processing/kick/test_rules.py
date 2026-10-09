"""Regression checks for kick direction, support foot and measured-range binning."""

import unittest

import numpy as np
from omegaconf import OmegaConf

from data_processing.kick.repair import physical_check
from data_processing.kick.repair import semantic_failures
from data_processing.parameter_coverage import summarize_split


class KickRulesTests(unittest.TestCase):
    def setUp(self):
        self.rules = OmegaConf.create(
            dict(
                minimum_forward_excursion_m=0.18,
                minimum_ankle_lift_m=0.1,
                maximum_lateral_ratio=0.65,
                maximum_root_excursion_m=0.65,
                maximum_support_foot_excursion_m=0.3,
                maximum_heading_change_rad=0.65,
            )
        )
        self.joints = np.zeros((60, 24, 3), dtype=np.float32)
        self.joints[:, 0, 2] = 1
        self.joints[:, 1, 1] = 0.15
        self.joints[:, 2, 1] = -0.15
        pulse = np.exp(-(((np.arange(60) - 24) / 5) ** 2))
        self.joints[:, 8, 0] = 0.6 * pulse
        self.joints[:, 8, 2] = 0.35 * pulse

    def test_forward_kick_passes(self):
        _, failures = physical_check(self.joints, 24, self.rules)
        self.assertEqual(failures, [])

    def test_side_or_round_kick_fails(self):
        self.joints[:, 8, 1] = self.joints[:, 8, 0]
        _, failures = physical_check(self.joints, 24, self.rules)
        self.assertIn("lateral_or_round_kick", failures)

    def test_support_foot_travel_fails(self):
        self.joints[:, 7, 0] = np.linspace(0, 0.6, 60)
        _, failures = physical_check(self.joints, 24, self.rules)
        self.assertIn("support_foot_excursion_m_above_limit", failures)

    def test_left_kick_in_context_fails(self):
        self.joints[:, 7] = self.joints[:, 8]
        _, failures = physical_check(self.joints, 24, self.rules)
        self.assertIn("left_kick_inside_crop", failures)

    def test_category_required(self):
        self.assertIn(
            "no_kick_category", semantic_failures(dict(proc_label="kick", act_cat=[]))
        )
        self.assertEqual(
            semantic_failures(
                dict(proc_label="kick right foot forward", act_cat=["kick"])
            ),
            [],
        )
        self.assertTrue(
            semantic_failures(dict(proc_label="right round kick", act_cat=["kick"]))
        )

    def test_bin_boundaries_and_independent_sources(self):
        records = [
            dict(key="first", family="same", quantity=1.5),
            dict(key="second", family="same", quantity=1.6),
            dict(key="third", family="other", quantity=2.5),
            dict(key="outside", family="extra", quantity=3),
        ]
        result = summarize_split(records, np.array([1.5, 2.0, 2.5]))
        self.assertEqual(result["inside_range"], 3)
        self.assertEqual(result["above_range"], 1)
        self.assertEqual(result["bins"][0]["samples"], 2)
        self.assertEqual(result["bins"][0]["families"], 1)
        self.assertEqual(result["bins"][1]["samples"], 1)


if __name__ == "__main__":
    unittest.main()
