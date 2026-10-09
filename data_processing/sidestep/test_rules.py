"""Direction and reflection checks that catch silent left/right corruption."""

from pathlib import Path
import unittest

import numpy as np
from omegaconf import OmegaConf
from scipy.spatial.transform import Rotation

from data_processing.sidestep.rules import BODY_SWAP
from data_processing.sidestep.rules import check_clip
from data_processing.sidestep.rules import mirror_body


class SidestepRulesTest(unittest.TestCase):
    def setUp(self):
        self.rules = OmegaConf.load(
            Path(__file__).parents[2] / "config/sidestep_repair.yaml"
        ).rules
        self.joints = np.zeros((31, 24, 3), dtype=np.float32)
        self.joints[:, :, 1] = np.linspace(0, 0.7, 31)[:, None]
        self.joints[:, 1, 1] += 0.1
        self.joints[:, 2, 1] -= 0.1
        separation = 0.1 + 0.4 * np.sin(np.linspace(0, np.pi, 31))
        self.joints[:, 7, 1] += separation / 2
        self.joints[:, 8, 1] -= separation / 2

    def test_mirror_is_an_involution(self):
        generator = np.random.default_rng(7)
        poses = generator.normal(size=(11, 66))
        translation = generator.normal(size=(11, 3))
        restored = mirror_body(*mirror_body(poses, translation))
        np.testing.assert_array_equal(restored[0], poses)
        np.testing.assert_array_equal(restored[1], translation)

    def test_mirror_rotations_have_correct_reflection(self):
        generator = np.random.default_rng(8)
        poses = generator.normal(size=(4, 66))
        reflected, _ = mirror_body(poses, np.zeros((4, 3)))
        original_matrices = (
            Rotation.from_rotvec(poses.reshape(-1, 3))
            .as_matrix()
            .reshape(4, 22, 3, 3)[:, BODY_SWAP]
        )
        mirrored_matrices = (
            Rotation.from_rotvec(reflected.reshape(-1, 3))
            .as_matrix()
            .reshape(4, 22, 3, 3)
        )
        reflection = np.diag([-1, 1, 1])
        np.testing.assert_allclose(
            mirrored_matrices, reflection @ original_matrices @ reflection, atol=1e-12
        )
        np.testing.assert_allclose(np.linalg.det(mirrored_matrices), 1, atol=1e-12)

    def test_left_passes_and_cannot_be_called_right(self):
        self.assertEqual(check_clip(self.joints, "left", self.rules)[1], [])
        self.assertIn(
            "wrong_direction_or_too_little_lateral_travel",
            check_clip(self.joints, "right", self.rules)[1],
        )

    def test_right_passes_after_reflection_and_side_swap(self):
        mirrored = self.joints[
            :,
            [
                0,
                2,
                1,
                3,
                5,
                4,
                6,
                8,
                7,
                9,
                11,
                10,
                12,
                14,
                13,
                15,
                17,
                16,
                19,
                18,
                21,
                20,
                23,
                22,
            ],
        ].copy()
        mirrored[:, :, 1] *= -1
        self.assertEqual(check_clip(mirrored, "right", self.rules)[1], [])

    def test_forward_walk_is_rejected(self):
        self.joints[:, :, 0] = np.linspace(0, 0.8, 31)[:, None]
        self.assertIn(
            "excess_forward_backward_travel",
            check_clip(self.joints, "left", self.rules)[1],
        )

    def test_cross_step_is_rejected(self):
        self.joints[15, 7, 1] = self.joints[15, 8, 1] - 0.1
        self.assertIn("crossing_feet", check_clip(self.joints, "left", self.rules)[1])


if __name__ == "__main__":
    unittest.main()
