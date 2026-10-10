"""Coordinate invariants and teacher-free residual integration contracts."""

import copy
import math
import unittest

import torch

from shared_motion.training.reach3d_adapter import ReachTargetResidual
from shared_motion.training.reach3d_geometry import wrist_positions_in_body_frame
from src.model.backbones.frankenmotion import TransformerDenoiser


class ReachGeometryTests(unittest.TestCase):
    def setUp(self):
        self.positions = torch.zeros(2, 4, 22, 3, dtype=torch.float64)
        self.positions[..., 1, 1] = 0.1
        self.positions[..., 2, 1] = -0.1
        self.positions[..., 20, :] = torch.tensor([0.4, 0.3, 0.2])
        self.positions[..., 21, :] = torch.tensor([0.5, -0.25, 0.1])

    def test_body_coordinates_and_rigid_transform_invariance(self):
        expected = self.positions[..., [20, 21], :]
        angle = 0.9
        rotation = torch.tensor(
            [
                [math.cos(angle), -math.sin(angle), 0],
                [math.sin(angle), math.cos(angle), 0],
                [0, 0, 1],
            ],
            dtype=torch.float64,
        )
        transformed = self.positions @ rotation.T + torch.tensor([2.0, -1.0, 0.8])
        torch.testing.assert_close(
            wrist_positions_in_body_frame(self.positions), expected
        )
        torch.testing.assert_close(wrist_positions_in_body_frame(transformed), expected)

    def test_mirror_preserves_forward_up_and_swaps_hand(self):
        mirrored = self.positions.clone()
        mirrored[..., 1] *= -1
        mirrored[..., [1, 2, 20, 21], :] = mirrored[..., [2, 1, 21, 20], :].clone()
        expected = wrist_positions_in_body_frame(self.positions)[..., [1, 0], :]
        expected = expected * torch.tensor([1, -1, 1])
        torch.testing.assert_close(wrist_positions_in_body_frame(mirrored), expected)

    def test_degenerate_frame_rejected_and_gradients_finite(self):
        with self.assertRaises(ValueError):
            wrist_positions_in_body_frame(torch.zeros_like(self.positions))
        positions = self.positions.clone().requires_grad_()
        wrist_positions_in_body_frame(positions).square().sum().backward()
        self.assertTrue(torch.isfinite(positions.grad).all())
        self.assertGreater(float(positions.grad[..., 20:22, :].abs().sum()), 0)


class ReachAdapterTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(82001)
        self.backbone = TransformerDenoiser(
            nfeats=613,
            tx_dim=512,
            latent_dim=32,
            ff_size=64,
            num_layers=2,
            num_heads=4,
            dropout=0.1,
            nb_registers=0,
        ).eval()
        self.original = copy.deepcopy(self.backbone)
        self.adapter = ReachTargetResidual(
            self.backbone, [0, 0, 0], [1, 1, 1], bottleneck=16
        )
        self.motion = torch.randn(2, 6, 613)
        self.timesteps = torch.tensor([20, 70])
        self.conditioning = {
            "mask": torch.ones(2, 6, dtype=torch.bool),
            "tx": {
                "x": torch.randn(2, 1, 512),
                "mask": torch.ones(2, 1, dtype=torch.bool),
            },
            "reach_target": {
                "hands": torch.tensor([0, 1]),
                "positions": torch.tensor([[0.4, 0.2, 0.1], [0.5, -0.2, 0.3]]),
            },
        }

    def tearDown(self):
        self.adapter.remove_hooks()

    def test_zero_residual_preserves_backbone_and_training_freezes_it(self):
        self.adapter.train()
        self.assertFalse(self.backbone.training)
        expected = self.original(self.motion, self.conditioning, self.timesteps)
        actual = self.adapter(self.motion, self.conditioning, self.timesteps)
        torch.testing.assert_close(actual, expected, atol=1e-6, rtol=1e-5)
        optimizer = torch.optim.Adam(
            [
                parameter
                for parameter in self.adapter.parameters()
                if parameter.requires_grad
            ],
            lr=1e-3,
        )
        actual.square().mean().backward()
        self.assertTrue(
            all(parameter.grad is None for parameter in self.backbone.parameters())
        )
        self.assertGreater(
            float(self.adapter.residuals[0][-1].weight.grad.abs().sum()), 0
        )
        optimizer.step()
        for name, state in self.backbone.state_dict().items():
            self.assertTrue(torch.equal(state, self.original.state_dict()[name]))

    def test_target_encoding_responds_to_all_axes_and_hand(self):
        positions = torch.zeros(4, 3)
        positions[1:] = torch.eye(3)
        hands = torch.zeros(4, dtype=torch.long)
        encoded = self.adapter.encode_targets(hands, positions)
        for axis in range(3):
            self.assertFalse(torch.equal(encoded[0], encoded[axis + 1]))
        opposite = self.adapter.encode_targets(torch.ones_like(hands), positions)
        self.assertFalse(torch.equal(encoded, opposite))

    def test_no_target_uses_backbone_and_bad_hand_is_rejected(self):
        conditioning = dict(self.conditioning)
        conditioning.pop("reach_target")
        expected = self.original(self.motion, conditioning, self.timesteps)
        actual = self.adapter(self.motion, conditioning, self.timesteps)
        torch.testing.assert_close(actual, expected)
        with self.assertRaises(ValueError):
            self.adapter.encode_targets(torch.tensor([2]), torch.zeros(1, 3))
        self.assertIsNone(self.adapter._condition)


if __name__ == "__main__":
    unittest.main()
