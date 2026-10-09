"""Meaningful regressions for the failure modes that polluted wave."""

import unittest
import numpy as np
from data_processing.wave.repair import canonical, chunks, waveform, ADMISSION
from data_processing.wave.rules import metrics, WAVE


class WaveSourceTests(unittest.TestCase):
    def positions(self):
        j = np.zeros((100, 24, 3), np.float32)
        j[:, 16] = [0, 0.2, 0.8]
        j[:, 17] = [0, -0.2, 0.8]
        j[:, 21] = [0, -0.3, 0.6]
        return j

    def test_wavy_walk_not_wave(self):
        self.assertIsNone(WAVE.search("walks forward in a wavy line"))
        self.assertIsNotNone(WAVE.search("waves with right hand"))

    def test_source_wrapper_removed_without_changing_take(self):
        self.assertEqual(
            canonical("MPIHDM05/MPI_HDM05/bk/HDM_bk_05-01_01_120_poses.npz"),
            "MPI_HDM05/bk/HDM_bk_05-01_01_120_poses",
        )

    def test_no_padding_or_gaps(self):
        self.assertEqual(chunks(0, 39), [])
        for length in [40, 120, 121, 239, 241, 1000]:
            windows = chunks(53, 53 + length)
            self.assertEqual(windows[0][0], 53)
            self.assertEqual(windows[-1][1], 53 + length)
            self.assertTrue(all(40 <= b - a <= 120 for a, b in windows))
            self.assertTrue(all(a[1] == b[0] for a, b in zip(windows, windows[1:])))

    def test_lateral_wave_vs_single_reach(self):
        j = self.positions()
        j[:, 21, 1] += 0.15 * np.sin(np.linspace(0, 6 * np.pi, len(j)))
        self.assertGreaterEqual(waveform(j)["lateral_reversals"], 4)
        j[:, 21, 1] = np.linspace(-0.3, 0.3, len(j))
        self.assertEqual(waveform(j)["lateral_reversals"], 0)
        j[:, 21, 1] = -0.3
        self.assertEqual(waveform(j)["lateral_reversals"], 0)

    def test_returning_to_start_does_not_hide_walking(self):
        j = self.positions()
        displacement = np.sin(np.linspace(0, np.pi, len(j)))
        j[:, :, 0] += displacement[:, None]
        self.assertGreater(
            metrics(j)["root_excursion_m"], ADMISSION["root_excursion_m"]
        )
        self.assertGreater(
            metrics(j)["foot_excursion_m"], ADMISSION["foot_excursion_m"]
        )

    def test_feet_moving_while_root_static_rejected(self):
        j = self.positions()
        j[:, 7, 0] = np.linspace(0, 0.3, len(j))
        self.assertEqual(metrics(j)["root_excursion_m"], 0)
        self.assertGreater(
            metrics(j)["foot_excursion_m"], ADMISSION["foot_excursion_m"]
        )


if __name__ == "__main__":
    unittest.main()
