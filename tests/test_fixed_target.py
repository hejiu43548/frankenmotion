import torch
from test_strike_residual_policy import ArmResidualTests
from shared_motion.training.fixed_target import (
    freeze_initial_target,
    body_basis,
    fixed_observations,
    apply_fixed_residual,
    fixed_metrics,
)


class FixedTargetTests(ArmResidualTests):
    def test_world_point_does_not_follow_body(self):
        joints = self.skeleton(self.motion)
        relative = torch.tensor([[0.5, -0.2, 0.4], [0.5, 0.2, 0.4]])
        target = freeze_initial_target(relative, joints[:, 0])
        local_at_end = (
            body_basis(joints[:, -1]).transpose(-1, -2)
            @ (target - joints[:, -1, 0])[..., None]
        ).squeeze(-1)
        self.assertGreater(float((local_at_end - relative).norm()), 0.1)
        reconstructed = joints[:, -1, 0] + (
            body_basis(joints[:, -1]) @ local_at_end[..., None]
        ).squeeze(-1)
        torch.testing.assert_close(reconstructed, target)

    def test_left_and_right_control_and_world_measurement(self):
        hands = torch.tensor([0, 1])
        joints = self.skeleton(self.motion)
        indices = torch.arange(2)
        target = joints[indices, self.frames, 20 + hands].clone()
        metrics = fixed_metrics(joints, target, hands, self.frames, "reach")
        torch.testing.assert_close(metrics["error_m"], torch.zeros(2))
        state = fixed_observations(
            self.motion, target, self.frames, hands, self.skeleton
        )
        self.assertEqual(state.shape, (2, 29))
        corrected = apply_fixed_residual(
            self.motion, torch.ones(2, 9), self.frames, hands, self.skeleton
        )
        after = self.skeleton(corrected)
        torch.testing.assert_close(after[0, :, 21], joints[0, :, 21])
        torch.testing.assert_close(after[1, :, 20], joints[1, :, 20])
        self.assertTrue(torch.equal(corrected[..., :4], self.motion[..., :4]))
        self.assertGreater(
            float((after[indices, self.frames, 20 + hands] - target).norm()), 0.01
        )
