"""Regressions for box collisions, punch handedness/style and speed timing."""

import unittest
import numpy as np
from data_processing.strike.rules import (
    semantic_reasons,
    signals,
    POLICY,
    left_punch_events,
)


class StrikeRulesTests(unittest.TestCase):
    def label(self, text, categories=("punch",)):
        return dict(proc_label=text, act_cat=list(categories))

    def test_box_object_not_punch(self):
        self.assertIn(
            "no_punch_act_cat",
            semantic_reasons(self.label("put down a box", ("place something",))),
        )

    def test_right_forward_jab(self):
        for text in [
            "right jab",
            "punch forward right arm",
            "right hand punch forward",
        ]:
            self.assertEqual(semantic_reasons(self.label(text)), [])

    def test_wrong_styles(self):
        for text in [
            "left punch",
            "right hook",
            "right uppercut",
            "right hand punch side",
            "right lob punch series",
            "backwards right punch",
            "punches with both hands",
        ]:
            self.assertTrue(semantic_reasons(self.label(text)), text)

    def test_mixed_kick(self):
        self.assertIn(
            "conflicting_action_category",
            semantic_reasons(self.label("right punch", ("punch", "kick"))),
        )

    def test_speed_units_and_retraction(self):
        j = np.zeros((60, 24, 3), np.float32)
        j[:, 1, 1] = 0.2
        j[:, 2, 1] = -0.2
        j[:, 17] = [0, -0.2, 1.4]
        j[:, 19] = [0.2, -0.2, 1.4]
        j[:, 21] = np.c_[np.arange(60) * 0.1, np.full(60, -0.2), np.full(60, 1.4)]
        np.testing.assert_allclose(signals(j)["world_speed"], 2, atol=1e-5)
        np.testing.assert_allclose(signals(j)["forward_speed"], 2, atol=1e-5)
        j[:, 21, 0] *= -1
        np.testing.assert_allclose(signals(j)["forward_speed"], -2, atol=1e-5)
        np.testing.assert_allclose(signals(j)["world_speed"], 2, atol=1e-5)

    def test_crop_options_fit_legacy_metric(self):
        for n in POLICY["crop_frame_options"]:
            self.assertTrue(40 <= n <= 60)
            for p in POLICY["peak_frame_options"]:
                self.assertTrue(16 <= p <= 33)
                self.assertTrue(p + 7 < n)

    def test_left_punch_outside_metric_window_is_still_contamination(self):
        j = np.zeros((60, 24, 3), np.float32)
        j[:, 1, 1] = 0.2
        j[:, 2, 1] = -0.2
        j[:, 16] = [0, 0.2, 1.4]
        j[:, 18] = [0.25, 0.2, 1.4]
        j[:, 20] = [0.1, 0.2, 1.4]
        j[5:10, 20, 0] = np.linspace(0.1, 0.6, 5)
        j[10:13, 20, 0] = 0.6
        j[13:18, 20, 0] = np.linspace(0.6, 0.1, 5)
        events = left_punch_events(j)
        self.assertTrue(events)
        self.assertLess(events[0]["frame"], 16)


if __name__ == "__main__":
    unittest.main()
