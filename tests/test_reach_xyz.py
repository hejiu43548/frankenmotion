"""XYZ command contract, gradients, mixed-task batching and legacy rejection."""

import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
from omegaconf import OmegaConf
import torch

from shared_motion.training.catalog import (
    ACTIVE_TASK_NAMES,
    COMMAND_RANGES,
    TASK_NAMES,
    measure_commands,
)
from shared_motion.training.data import MotionDataset
from shared_motion.training.geometry import Skeleton
from shared_motion.training.losses import LossRecipe
from shared_motion.training.model import build_model, file_sha256
from shared_motion.training.reach import REACH_REVISION, command_vectors, target_xyz
from shared_motion.training.runner import checkpoint_compatible, scan, audit_scans
from tests.test_staged_training import composed
from tests.training_fixtures import make_data


class ReachXYZTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.data = make_data(cls.root / "data")
        cls.skeleton = Skeleton(cls.data / "skeleton.npz")
        for split in ["train", "val"]:
            path = cls.data / f"{split}.json"
            rows = json.loads(path.read_text())
            for record in rows:
                if record["task"] != "reach":
                    continue
                with np.load(record["cache"]) as archive:
                    arrays = {name: archive[name].copy() for name in archive.files}
                target = target_xyz(
                    cls.skeleton(torch.from_numpy(arrays["motion"])[None])
                )[0].numpy()
                arrays["quantity"] = target
                np.savez(record["cache"], **arrays)
                record.update(
                    semantic_revision=REACH_REVISION,
                    target_xyz_m=target.tolist(),
                    quantity=target.tolist(),
                    real_frames=120,
                    pad_frames=0,
                    cache_sha256=file_sha256(record["cache"]),
                )
            path.write_text(json.dumps(rows))

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def dataset(self):
        return MotionDataset(self.data / "train.json", "train", ACTIVE_TASK_NAMES)

    def model(self, kind):
        config = composed(2, kind)
        config.backbone = dict(_target_="tests.training_fixtures.toy_bundle")
        if kind == "with_root":
            config.controller.bottleneck = 4
            config.controller.embedding_width = 4
        else:
            config.controller.width = 16
        return build_model(config, "task")

    def test_scalar_and_retired_point_rejected(self):
        with self.assertRaisesRegex(ValueError, "XYZ"):
            command_vectors(torch.tensor([0.4]), torch.tensor([1]))
        with self.assertRaisesRegex(ValueError, "merged"):
            command_vectors(torch.tensor([[0.4, 0, 0]]), torch.tensor([14]))
        with self.assertRaisesRegex(ValueError, "merged"):
            MotionDataset(self.data / "train.json", "train", ["point"])

    def test_mixed_batch_and_all_xyz_axes_reach_both_encoders(self):
        dataset = self.dataset()
        batch = dataset.batch(
            [dataset.groups[TASK_NAMES.index(name)][0] for name in ACTIVE_TASK_NAMES],
            "cpu",
        )
        self.assertEqual(batch["quantity"].shape, (19, 3))
        for kind in ["with_root", "without_root"]:
            model = self.model(kind)
            commands = batch["quantity"].clone().requires_grad_()
            if kind == "with_root":
                features = model.denoiser.encoded(batch["task"], commands)
            else:
                control = model.requested_root(batch, commands, self.skeleton)
                features = model.unified_features(batch, commands, control, True)
                self.assertEqual(features.shape[-1], 58)
            gradient = torch.autograd.grad(features.square().sum(), commands)[0]
            self.assertTrue((gradient[1].abs() > 0).all(), (kind, gradient[1]))
            nonreach = batch["task"] != 1
            self.assertTrue((gradient[nonreach, 1:] == 0).all())
            raw = model.sample(
                batch, batch["quantity"], self.skeleton, list(range(19)), 2
            )
            measured = measure_commands(
                self.skeleton, raw, batch["task"], batch["lengths"]
            )
            self.assertEqual(measured.shape, (19, 3))
            self.assertTrue(torch.isfinite(measured).all())

    def test_final_target_is_same_time_xyz_not_axiswise_maxima(self):
        positions = torch.zeros(1, 20, 24, 3)
        positions[:, :, 1, 1] = 0.1
        positions[:, :, 2, 1] = -0.1
        positions[:, 0, 21] = torch.tensor([1.0, 2.0, 3.0])
        positions[:, -5:, 21] = torch.tensor([0.4, -0.2, 0.3])
        torch.testing.assert_close(
            target_xyz(positions), torch.tensor([[0.4, -0.2, 0.3]])
        )
        offset = torch.tensor([3.0, -2.0, 5.0])
        torch.testing.assert_close(
            target_xyz(positions + offset), target_xyz(positions)
        )

    def test_xyz_supervised_loss_and_finite_gradients_both_controllers(self):
        dataset = self.dataset()
        batch = dataset.batch([dataset.groups[1][0], dataset.groups[10][0]], "cpu")
        recipe = LossRecipe("supervised")
        for kind in ["with_root", "without_root"]:
            model = self.model(kind)
            loss, metrics = recipe.supervised(
                model, self.skeleton, batch, training=False
            )
            loss.backward()
            self.assertTrue(torch.isfinite(loss))
            self.assertTrue(
                all(
                    torch.isfinite(parameter.grad).all()
                    for parameter in model.parameters()
                    if parameter.grad is not None
                )
            )

    def test_xyz_validation_scan_and_recomputation(self):
        dataset = MotionDataset(self.data / "val.json", "val", ["reach", "walk"])
        config = composed(2)
        config.data.tasks = ["reach", "walk"]
        config.runtime.device = "cpu"
        config.validation.points = 2
        config.validation.ddim_steps = 2
        folder = self.root / "scan_validation"
        result = scan(
            self.model("with_root"), self.skeleton, dataset, config, folder, 0
        )
        self.assertEqual(result["tasks"][0]["command_dimension"], 3)
        self.assertEqual(result["tasks"][0]["metric"], "XYZ Euclidean distance (m)")
        (folder / "scans").mkdir(exist_ok=True)
        (folder / "scans/scan_000000.json").write_text(json.dumps(result))
        self.assertEqual(audit_scans(folder, self.skeleton, "cpu")["rollouts"], 3)

    def test_old_checkpoint_policy_rejected(self):
        model = self.model("with_root")
        model.backbone_configuration = {}
        checkpoint = dict(
            controller_kind=model.kind,
            backbone_sha256=model.backbone_sha256,
            tasks=["reach"],
            task_policy={"revision": "walking_turn_v2"},
        )
        with self.assertRaisesRegex(ValueError, "semantics differ"):
            checkpoint_compatible(checkpoint, model, ["reach"])


if __name__ == "__main__":
    unittest.main()
