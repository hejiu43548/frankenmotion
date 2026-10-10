"""Numerical regressions for short gait-phase bridges and their time grids."""

from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
from g1_demo2_gait import hermite, phase_resample


class Demo2GaitPhaseTests(unittest.TestCase):
    def test_hermite_preserves_positions_and_endpoint_velocities(self):
        first = np.array([0.0, 1.0])
        last = np.array([0.3, 0.8])
        first_velocity = np.array([1.2, -0.1])
        last_velocity = np.array([0.7, 0.2])
        duration = 0.2
        epsilon = 1e-6
        samples = hermite(
            first,
            last,
            first_velocity,
            last_velocity,
            np.array([0.0, epsilon, 1 - epsilon, 1.0]),
            duration,
        )
        np.testing.assert_array_equal(samples[0], first)
        np.testing.assert_array_equal(samples[-1], last)
        np.testing.assert_allclose(
            (samples[1] - samples[0]) / (duration * epsilon), first_velocity, atol=1e-5
        )
        np.testing.assert_allclose(
            (samples[-1] - samples[-2]) / (duration * epsilon), last_velocity, atol=1e-5
        )

    def test_resample_clamps_roundoff_at_3_point_3_seconds(self):
        states = np.zeros((67, 36))
        states[:, 0] = np.arange(67) / 20
        states[:, 3] = 1.0
        result = phase_resample(states, 1.0)
        self.assertEqual(result.shape, (166, 36))
        self.assertAlmostEqual(result[-1, 0], 3.3)
        np.testing.assert_allclose(np.linalg.norm(result[:, 3:7], axis=1), 1.0)
        self.assertTrue(np.isfinite(result).all())


if __name__ == "__main__":
    unittest.main()
