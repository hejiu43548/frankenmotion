"""Exercise denoising training, source-free sampling and residual-only states."""

import unittest

import torch

from shared_motion.training.reach3d_model import ReachDiffusion
from src.model.backbones.frankenmotion import TransformerDenoiser


class ReachDiffusionTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(84001)
        backbone = TransformerDenoiser(
            nfeats=613,
            tx_dim=512,
            latent_dim=32,
            ff_size=64,
            num_layers=2,
            num_heads=4,
            dropout=0.1,
            nb_registers=0,
        )
        bundle = dict(
            denoiser=backbone,
            sha256="synthetic-test-only",
            mean=torch.randn(613),
            std=torch.rand(613) + 0.1,
            text_mean=torch.zeros(512),
            text_std=torch.ones(512),
            alphas=torch.linspace(0.99, 0.01, 10),
        )
        self.model = ReachDiffusion(bundle, [0, 0, 0], [1, 1, 1], bottleneck=16)
        self.batch = dict(
            motion=torch.randn(2, 6, 205),
            local=torch.randn(2, 6, 408),
            local_mask=torch.ones(2, 6, 408, dtype=torch.bool),
            tx=torch.randn(2, 512),
            mask=torch.tensor([[True] * 6, [True] * 4 + [False] * 2]),
            hands=torch.tensor([0, 1]),
            positions=torch.tensor([[0.3, 0.2, 0.1], [0.4, -0.2, 0.0]]),
        )

    def tearDown(self):
        self.model.denoiser.remove_hooks()

    def sample(self, enabled=True, positions=None):
        return self.model.sample(
            self.batch["tx"],
            self.batch["local"],
            self.batch["local_mask"],
            self.batch["mask"],
            self.batch["hands"],
            self.batch["positions"] if positions is None else positions,
            seeds=[51, 52],
            steps=4,
            enabled=enabled,
        )

    def test_denoising_step_and_padding_do_not_update_base(self):
        self.model.train()
        before = {
            name: value.clone()
            for name, value in self.model.denoiser.base.state_dict().items()
        }
        loss, metrics = self.model.supervised_loss(
            self.batch, torch.Generator().manual_seed(91)
        )
        altered = dict(self.batch)
        altered["motion"] = self.batch["motion"].clone()
        altered["motion"][~self.batch["mask"]] = 10000
        repeated, _ = self.model.supervised_loss(
            altered, torch.Generator().manual_seed(91)
        )
        torch.testing.assert_close(loss, repeated)
        self.assertTrue(torch.isfinite(loss))
        optimizer = torch.optim.Adam(
            [
                parameter
                for parameter in self.model.parameters()
                if parameter.requires_grad
            ],
            lr=1e-3,
        )
        loss.backward()
        optimizer.step()
        for name, value in self.model.denoiser.base.state_dict().items():
            self.assertTrue(torch.equal(before[name], value))
        self.assertTrue(
            all(
                parameter.grad is None
                for parameter in self.model.denoiser.base.parameters()
            )
        )
        self.assertGreater(float(metrics["reconstruction"]), 0)

    def test_sampling_repeats_and_new_heads_change_generation(self):
        self.model.eval()
        baseline = self.sample(enabled=False)
        torch.testing.assert_close(self.sample(), baseline)
        self.assertTrue(torch.equal(self.sample(), self.sample()))
        self.assertTrue(
            torch.equal(
                baseline[~self.batch["mask"]],
                torch.zeros_like(baseline[~self.batch["mask"]]),
            )
        )
        with torch.no_grad():
            for residual in self.model.denoiser.residuals:
                residual[-1].weight.normal_(std=0.01)
        changed = self.sample()
        other_target = self.sample(positions=self.batch["positions"] + 0.5)
        self.assertGreater(float((changed - baseline).abs().max()), 1e-5)
        self.assertGreater(float((changed - other_target).abs().max()), 1e-5)
        torch.testing.assert_close(self.sample(enabled=False), baseline)

    def test_state_round_trip_and_base_injection_rejected(self):
        state = self.model.adapter_state()
        self.assertFalse(any(name.startswith("base.") for name in state))
        expected = self.sample()
        with torch.no_grad():
            self.model.denoiser.residuals[0][-1].bias.fill_(0.2)
        self.model.load_adapter(state)
        torch.testing.assert_close(self.sample(), expected)
        polluted = dict(state)
        polluted["base.skel_embedding.weight"] = (
            self.model.denoiser.base.skel_embedding.weight
        )
        with self.assertRaises(ValueError):
            self.model.load_adapter(polluted)
        invalid = dict(state)
        invalid["target_scale"] = torch.zeros(3)
        with self.assertRaises(ValueError):
            self.model.load_adapter(invalid)

    def test_invalid_masks_and_sampling_inputs_rejected(self):
        invalid = dict(self.batch)
        invalid["mask"] = self.batch["mask"].clone()
        invalid["mask"][0, 1] = False
        with self.assertRaises(ValueError):
            self.model.supervised_loss(invalid, torch.Generator())
        with self.assertRaises(ValueError):
            self.model.sample(
                self.batch["tx"],
                self.batch["local"],
                self.batch["local_mask"],
                self.batch["mask"],
                self.batch["hands"],
                self.batch["positions"],
                seeds=[1],
                steps=4,
            )


if __name__ == "__main__":
    unittest.main()
