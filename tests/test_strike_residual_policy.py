import tempfile
from pathlib import Path
import unittest

import numpy as np
import torch

from shared_motion.training.geometry import Skeleton
from shared_motion.training.strike_residual_policy import (
    apply_residual,
    observations,
    StrikeResidualPolicy,
)
from src.tools.geometry import matrix_to_rotation_6d, axis_angle_rotation
from src.tools.smplrifke_feats import smplrifkefeats_to_smpldata


class ArmResidualTests(unittest.TestCase):
    def setUp(self):
        parents = np.array(
            [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
        )
        positions = np.zeros((38, 3), np.float32)
        offsets = {
            1: [0, 0.1, 0],
            2: [0, -0.1, 0],
            3: [0, 0, 0.15],
            4: [0, 0, -0.4],
            5: [0, 0, -0.4],
            6: [0, 0, 0.15],
            7: [0, 0, -0.4],
            8: [0, 0, -0.4],
            9: [0, 0, 0.15],
            10: [0.1, 0, 0],
            11: [0.1, 0, 0],
            12: [0, 0, 0.15],
            13: [0, 0.08, 0.1],
            14: [0, -0.08, 0.1],
            15: [0, 0, 0.15],
            16: [0, 0.12, 0],
            17: [0, -0.12, 0],
            18: [0, 0.25, 0],
            19: [0, -0.25, 0],
            20: [0, 0.25, 0],
            21: [0, -0.25, 0],
        }
        for joint_index in range(1, 22):
            positions[joint_index] = (
                positions[parents[joint_index]] + offsets[joint_index]
            )
        positions[22] = positions[20] + [0, 0.1, 0]
        positions[37] = positions[21] + [0, -0.1, 0]
        self.directory = tempfile.TemporaryDirectory()
        path = Path(self.directory.name) / "skeleton.npz"
        np.savez(path, J=positions, parents=parents, height=1.0)
        self.skeleton = Skeleton(path)
        self.motion = torch.zeros(2, 30, 205)
        self.motion[..., 0] = 1
        self.motion[..., 1] = 0.01
        self.motion[..., 3] = 0.02
        rotations = torch.eye(3).repeat(2, 30, 22, 1, 1)
        rotations[:, :, 0] = axis_angle_rotation(
            "Y", torch.tensor(0.2)
        ) @ axis_angle_rotation("X", torch.tensor(0.1))
        self.motion[..., 4:136] = matrix_to_rotation_6d(rotations).flatten(-2)
        self.frames = torch.tensor([15, 15])

    def tearDown(self):
        self.directory.cleanup()

    def test_zero_action_preserves_fk_and_root(self):
        corrected = apply_residual(
            self.motion, torch.zeros(2, 9), self.frames, self.skeleton
        )
        torch.testing.assert_close(self.skeleton(corrected), self.skeleton(self.motion))
        self.assertTrue(torch.equal(corrected[..., :4], self.motion[..., :4]))
        state = observations(self.motion, torch.zeros(2, 3), self.frames, self.skeleton)
        self.assertEqual(state.shape, (2, 27))
        torch.testing.assert_close(
            StrikeResidualPolicy().mean(state), torch.zeros(2, 9)
        )

    def test_residual_changes_arm_only_with_consistent_export(self):
        corrected = apply_residual(
            self.motion, torch.ones(2, 9), self.frames, self.skeleton
        )
        original_joints = self.skeleton(self.motion)
        corrected_joints = self.skeleton(corrected)
        torch.testing.assert_close(
            corrected_joints[:, :, :14], original_joints[:, :, :14]
        )
        torch.testing.assert_close(corrected_joints[:, 0], original_joints[:, 0])
        self.assertGreater(
            float((corrected_joints[:, 15, 21] - original_joints[:, 15, 21]).norm()),
            0.01,
        )
        decoded = smplrifkefeats_to_smpldata(corrected[0])["joints"]
        side = decoded[0, 1, :2] - decoded[0, 2, :2]
        heading = torch.atan2(side[1], side[0]) - torch.pi / 2
        canonical = (axis_angle_rotation("Z", -heading) @ decoded[..., None]).squeeze(
            -1
        )
        torch.testing.assert_close(canonical, corrected_joints[0], atol=2e-5, rtol=1e-5)


if __name__ == "__main__":
    unittest.main()
