import ast
import copy
import json
from pathlib import Path
import tempfile
import unittest

from hydra import compose, initialize_config_dir
import numpy as np
from omegaconf import OmegaConf
import torch

from shared_motion.adapter.network import SharedCommands
from shared_motion.training.adapters import RootControl, TaskControl
from shared_motion.training.catalog import COMMAND_RANGES, TASK_NAMES, measure
from shared_motion.training.data import MotionDataset, StatefulSampler, assert_disjoint
from shared_motion.training.geometry import Skeleton
from shared_motion.training.model import (
    build_model,
    file_sha256,
    load_official_backbone,
)
from shared_motion.training.runner import run
from tests.training_fixtures import ToyBackbone, make_data, toy_bundle

REPOSITORY = Path(__file__).resolve().parents[1]


def composed(stage, controller="with_root", loss=None):
    overrides = [f"stage=stage{stage}", f"controller={controller}"]
    if loss:
        overrides.append(f"loss={loss}")
    with initialize_config_dir(
        config_dir=str(REPOSITORY / "config"), version_base="1.3"
    ):
        return compose(config_name="train", overrides=overrides)


def equal_nested(left, right):
    if torch.is_tensor(left):
        return torch.equal(left, right)
    if isinstance(left, dict):
        return left.keys() == right.keys() and all(
            equal_nested(left[name], right[name]) for name in left
        )
    if isinstance(left, (tuple, list)):
        return len(left) == len(right) and all(
            equal_nested(first, second) for first, second in zip(left, right)
        )
    if isinstance(left, np.ndarray):
        return np.array_equal(left, right)
    return left == right


class StagedTrainingTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        cls.root = Path(cls.temporary.name)
        cls.data = make_data(cls.root / "data")

    @classmethod
    def tearDownClass(cls):
        cls.temporary.cleanup()

    def config(self, stage, controller, name, initial=None, loss=None):
        config = composed(stage, controller, loss)
        config.backbone = dict(_target_="tests.training_fixtures.toy_bundle")
        if controller == "with_root":
            config.controller.bottleneck = 4
            config.controller.embedding_width = 4
        else:
            config.controller.width = 16
        config.initial = initial
        config.data.train_manifest = str(self.data / "train.json")
        config.data.val_manifest = str(self.data / "val.json")
        config.data.skeleton = str(self.data / "skeleton.npz")
        config.data.path_root = str(self.data)
        config.runtime.device = "cpu"
        config.runtime.threads = 1
        config.runtime.log_every = 1
        config.runtime.checkpoint_every = 1
        config.paths.report = str(self.root / name / "reports")
        config.paths.artifacts = str(self.root / name / "artifacts")
        config.stage.epochs = 1
        config.stage.steps = 3
        config.stage.batch_size = 20
        config.stage.accumulate = 1
        config.stage.eval_every = 2
        config.validation.points = 2
        config.validation.ddim_steps = 2
        config.validation.batch_size = 20
        OmegaConf.update(config.loss, "ddim_steps", 2, force_add=True)
        return config

    def test_hydra_all_combinations(self):
        for controller in ["with_root", "without_root"]:
            for stage in [1, 2, 3]:
                config = composed(stage, controller)
                self.assertEqual(len(config.data.tasks), 20)
                self.assertNotIn("spin", config.data.tasks)
                self.assertEqual(config.stage.index, stage)
            for loss in ["free_generation", "main_replay", "command_only"]:
                self.assertEqual(composed(3, controller, loss).loss.mode, "free")

    def test_parameter_contracts_and_zero_initialization(self):
        bundle = toy_bundle()
        root = RootControl(bundle["denoiser"])
        self.assertEqual(
            sum(
                parameter.numel()
                for parameter in root.parameters()
                if parameter.requires_grad
            ),
            593536,
        )
        task = TaskControl(root, COMMAND_RANGES)
        self.assertEqual(
            sum(
                parameter.numel()
                for parameter in task.parameters()
                if parameter.requires_grad
            ),
            597888,
        )
        self.assertEqual(
            sum(parameter.numel() for parameter in SharedCommands(1024).parameters()),
            8664269,
        )
        for controller in ["with_root", "without_root"]:
            config = self.config(2, controller, "unused")
            model = build_model(config, "task").eval()
            dataset = MotionDataset(self.data / "train.json", "train", TASK_NAMES)
            batch = dataset.batch(list(range(20)), "cpu")
            skeleton = Skeleton(self.data / "skeleton.npz")
            commands = batch["quantity"]
            with torch.no_grad():
                generated = model.sample(batch, commands, skeleton, list(range(20)), 2)
                baseline = model.sample(
                    batch, commands, skeleton, list(range(20)), 2, "backbone"
                )
                self.assertTrue(torch.equal(generated, baseline))
                poisoned = dict(
                    batch,
                    motion=torch.full_like(batch["motion"], float("nan")),
                    quantity=torch.full_like(batch["quantity"], float("nan")),
                )
                self.assertTrue(
                    torch.equal(
                        generated,
                        model.sample(poisoned, commands, skeleton, list(range(20)), 2),
                    )
                )
            self.assertEqual(
                measure(skeleton, generated, batch["task"], batch["lengths"]).shape,
                (20,),
            )

    def test_manifest_missing_tasks_and_leakage(self):
        records = json.loads((self.data / "train.json").read_text())
        incomplete = self.data / "missing.json"
        incomplete.write_text(json.dumps(records[:-1]))
        with self.assertRaisesRegex(ValueError, "Missing train task"):
            MotionDataset(incomplete, "train", TASK_NAMES)
        dataset = MotionDataset(self.data / "train.json", "train", TASK_NAMES)
        with self.assertRaisesRegex(ValueError, "leakage"):
            assert_disjoint(dataset, dataset)
        sampler = StatefulSampler(dataset, 31)
        first = sampler.batches(7)
        state = sampler.state_dict()
        expected = sampler.batches(33)
        restored = StatefulSampler(dataset, 0)
        restored.load_state_dict(state)
        self.assertEqual(expected, restored.batches(33))
        indices = first[0] + expected[0]
        counts = {
            name: sum(dataset.rows[index]["task"] == name for index in indices)
            for name in TASK_NAMES
        }
        self.assertEqual(set(counts.values()), {2})

    def test_official_loader_hash_and_strict_weights(self):
        bundle = toy_bundle()
        state = {
            "denoiser." + name: value
            for name, value in bundle["denoiser"].state_dict().items()
        }
        for name, prefix in [
            ("mean", "motion_normalizer.mean"),
            ("std", "motion_normalizer.std"),
            ("text_mean", "text_normalizer.mean"),
            ("text_std", "text_normalizer.std"),
        ]:
            state[prefix] = bundle[name]
        path = self.root / "fake_official.pt"
        torch.save(dict(state_dict=state), path)
        with self.assertRaisesRegex(ValueError, "SHA256"):
            load_official_backbone(
                path,
                ToyBackbone(),
                lambda steps: torch.zeros(steps),
                expected_sha256="wrong",
            )
        loaded = load_official_backbone(
            path,
            ToyBackbone(),
            lambda steps: torch.zeros(steps),
            expected_sha256=file_sha256(path),
        )
        self.assertTrue(
            equal_nested(
                loaded["denoiser"].state_dict(), bundle["denoiser"].state_dict()
            )
        )

    def test_stage_pipelines_resume_and_loss_switches(self):
        for controller in ["with_root", "without_root"]:
            root_config = self.config(1, controller, f"{controller}_root")
            run(root_config)
            parent = str(Path(root_config.paths.artifacts) / "best.pt")
            for stage in [2, 3]:
                full = self.config(
                    stage, controller, f"{controller}_stage{stage}", parent
                )
                run(full)
                replay = self.config(
                    stage, controller, f"{controller}_replay{stage}", parent
                )
                replay.runtime.stop_after = 1
                run(replay)
                replay.runtime.resume = str(Path(replay.paths.artifacts) / "latest.pt")
                replay.runtime.stop_after = None
                run(replay)
                full_state = torch.load(
                    Path(full.paths.artifacts) / "latest.pt", weights_only=False
                )
                replay_state = torch.load(
                    Path(replay.paths.artifacts) / "latest.pt", weights_only=False
                )
                for field in ["adapter", "optimizer", "sampler", "rng", "step"]:
                    self.assertTrue(
                        equal_nested(full_state[field], replay_state[field]),
                        (controller, stage, field),
                    )
                if stage == 3:
                    break
                parent = str(Path(full.paths.artifacts) / "best.pt")
            for loss_name in ["main_replay", "command_only"]:
                config = self.config(
                    3, controller, f"{controller}_{loss_name}", parent, loss_name
                )
                config.stage.steps = 1
                run(config)
                audit = json.loads(
                    (Path(config.paths.report) / "completion_audit.json").read_text()
                )
                self.assertTrue(audit["frozen_parameters_verified"])

    def test_root_resume_mid_epoch_and_changed_data(self):
        for controller in ["with_root", "without_root"]:
            full = self.config(1, controller, f"{controller}_root_continuous")
            full.stage.epochs = 2
            full.stage.batch_size = 8
            full.stage.accumulate = 2
            run(full)
            replay = self.config(1, controller, f"{controller}_root_mid_epoch")
            replay.stage.epochs = 2
            replay.stage.batch_size = 8
            replay.stage.accumulate = 2
            replay.runtime.stop_after = 1
            run(replay)
            replay.runtime.stop_after = None
            replay.runtime.resume = str(Path(replay.paths.artifacts) / "latest.pt")
            run(replay)
            first = torch.load(
                Path(full.paths.artifacts) / "latest.pt", weights_only=False
            )
            second = torch.load(
                Path(replay.paths.artifacts) / "latest.pt", weights_only=False
            )
            for field in ["adapter", "optimizer", "sampler", "rng"]:
                self.assertTrue(
                    equal_nested(first[field], second[field]), (controller, field)
                )
            path = self.data / "train_raise_hand.npz"
            original = path.read_bytes()
            with np.load(path) as archive:
                changed = {name: archive[name].copy() for name in archive.files}
            changed["quantity"] = changed["quantity"] + 0.01
            np.savez(path, **changed)
            try:
                with self.assertRaisesRegex(ValueError, "Resume protocol differs"):
                    run(replay)
            finally:
                path.write_bytes(original)
            skeleton_path = self.data / "skeleton.npz"
            original_skeleton = skeleton_path.read_bytes()
            with np.load(skeleton_path) as archive:
                changed_skeleton = {
                    name: archive[name].copy() for name in archive.files
                }
            changed_skeleton["height"] = changed_skeleton["height"] + 0.01
            np.savez(skeleton_path, **changed_skeleton)
            try:
                with self.assertRaisesRegex(ValueError, "Resume protocol differs"):
                    run(replay)
            finally:
                skeleton_path.write_bytes(original_skeleton)

    def test_new_python_style_ast(self):
        paths = list((REPOSITORY / "shared_motion/training").glob("*.py")) + [
            REPOSITORY / "scripts/train.py",
            REPOSITORY / "scripts/infer.py",
        ]
        for path in paths:
            module = ast.parse(path.read_text())
            for node in ast.walk(module):
                if isinstance(node, ast.Name):
                    self.assertFalse(
                        len(node.id) == 1 and node.id != "_", (path, node.id)
                    )
                if isinstance(node, ast.arg):
                    self.assertFalse(
                        len(node.arg) == 1 and node.arg != "_", (path, node.arg)
                    )
                if isinstance(node, ast.Import):
                    self.assertEqual(len(node.names), 1, path)


if __name__ == "__main__":
    unittest.main()
