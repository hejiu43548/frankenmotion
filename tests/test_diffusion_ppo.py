import unittest

import torch

from shared_motion.training.diffusion_ppo import gaussian_log_ratio, clipped_policy_loss


class DiffusionProbabilityTests(unittest.TestCase):
    def test_joint_log_ratio_matches_normal_and_ignores_padding(self):
        torch.manual_seed(54)
        action = torch.randn(2, 4, 3, dtype=torch.float64)
        old_mean = torch.randn_like(action)
        mean = torch.randn_like(action).requires_grad_()
        variance = torch.tensor(0.3, dtype=torch.float64)
        mask = torch.tensor([[True, True, False, False], [True] * 4])
        actual = gaussian_log_ratio(action, mean, old_mean, variance, mask)
        new_log = torch.distributions.Normal(mean, variance.sqrt()).log_prob(action)
        old_log = torch.distributions.Normal(old_mean, variance.sqrt()).log_prob(action)
        expected = ((new_log - old_log) * mask[..., None]).sum((1, 2))
        torch.testing.assert_close(actual, expected)
        actual.sum().backward()
        self.assertEqual(float(mean.grad[~mask].abs().sum()), 0)
        torch.testing.assert_close(
            gaussian_log_ratio(action, old_mean, old_mean, variance, mask),
            torch.zeros(2, dtype=torch.float64),
        )

    def test_clipping_stops_favorable_oversized_updates(self):
        log_ratio = torch.tensor([0.5, -0.5], requires_grad=True)
        loss, fraction = clipped_policy_loss(log_ratio, torch.tensor([1.0, -1.0]))
        loss.backward()
        torch.testing.assert_close(log_ratio.grad, torch.zeros(2))
        self.assertEqual(float(fraction), 1)


if __name__ == "__main__":
    unittest.main()
