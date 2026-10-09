"""Regression checks for moving turns versus stationary endpoint turning."""

import copy
import json
from pathlib import Path
import tempfile
import unittest

import torch

from shared_motion.training.catalog import (
    COMMAND_RANGES,
    TASK_NAMES,
    command_error,
    measure,
)
from shared_motion.training.data import MotionDataset
from shared_motion.training.geometry import Skeleton
from shared_motion.training.model import build_model
from shared_motion.training.runner import checkpoint_compatible
from shared_motion.training.turn import TURN_NATIVE_SPEED, sample_commands, scan_indices
from shared_motion.training.turn_data import check_admission, replace_turn_rows
from tests.test_staged_training import composed
from tests.training_fixtures import make_data


class WalkingTurnTest(unittest.TestCase):
    def test_signed_angles_root_conditions_and_directional_text(self):
        with tempfile.TemporaryDirectory() as directory:
            root = make_data(directory)
            data = MotionDataset(root / "train.json", "train", TASK_NAMES)
            turn_index = TASK_NAMES.index("turn")
            commands = torch.tensor([-3.4, 3.4])
            indices = scan_indices(data, turn_index, commands, turn_index)
            self.assertEqual(
                [data.rows[index]["direction"] for index in indices], ["left", "right"]
            )
            batch = data.batch(indices, "cpu")
            skeleton = Skeleton(root / "skeleton.npz")
            motion = batch["motion"].clone()
            motion[:, :, 3] = -commands[:, None] / 119
            measured = measure(skeleton, motion, batch["task"], batch["lengths"])
            self.assertTrue(torch.allclose(measured, commands, atol=1e-5))
            self.assertAlmostEqual(
                float(command_error(commands, commands.flip(0), batch["task"])[0]),
                -6.8,
                places=5,
            )
            for _ in range(5):
                sampled = sample_commands(batch, COMMAND_RANGES, turn_index)
                self.assertTrue(torch.equal(sampled.sign(), commands.sign()))
                self.assertTrue((sampled.abs() >= 0.35).all())
                self.assertTrue((sampled.abs() <= 3.5).all())
            for controller in ["with_root", "without_root"]:
                config = composed(2, controller)
                config.backbone = dict(_target_="tests.training_fixtures.toy_bundle")
                model = build_model(config, "task")
                control = model.requested_root(batch, commands, skeleton)
                self.assertTrue(
                    torch.allclose(
                        control[:, :, 0] * 3, torch.full((2, 120), TURN_NATIVE_SPEED)
                    )
                )
                self.assertTrue(
                    torch.allclose(control[:, 0, 1] * torch.pi, -commands / (119 / 20))
                )
                if controller == "without_root":
                    features = model.unified_features(batch, commands, control, True)
                    self.assertTrue(torch.allclose(features[:, 0, 16], commands / 3.5))
                    self.assertTrue(torch.equal(features[:, :, 18:22], control))
                legacy = dict(
                    controller_kind=model.kind,
                    backbone_sha256=model.backbone_sha256,
                    tasks=TASK_NAMES,
                )
                with self.assertRaisesRegex(ValueError, "task semantics differ"):
                    checkpoint_compatible(legacy, model, TASK_NAMES)

    def test_reject_legacy_sources_and_preserve_other_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = make_data(directory)
            rows = json.loads((root / "train.json").read_text())
            old = copy.deepcopy(rows)
            next(row for row in old if row["task"] == "turn").pop(
                "turn_source_revision"
            )
            path = root / "legacy.json"
            path.write_text(json.dumps(old))
            with self.assertRaisesRegex(ValueError, "Unverified/legacy turn source"):
                MotionDataset(path, "train", TASK_NAMES)
            with self.assertRaisesRegex(ValueError, "Unverified/legacy turn source"):
                MotionDataset(path, "train", TASK_NAMES, deduplicate=True)
            damaged = copy.deepcopy(rows)
            next(row for row in damaged if row["task"] == "turn")[
                "cache_sha256"
            ] = "wrong"
            path.write_text(json.dumps(damaged))
            with self.assertRaisesRegex(ValueError, "cache hash"):
                MotionDataset(path, "train", TASK_NAMES)
        nonturn = dict(task="squat", split="train", extra={"keep": "verbatim"})
        old_turn = dict(task="turn", split="train")
        replacement = [dict(task="turn", split="train", id="moving")]
        repaired = replace_turn_rows([old_turn, nonturn], replacement, "train")
        self.assertIs(repaired[0], nonturn)
        self.assertEqual(repaired[1:], replacement)
        with self.assertRaisesRegex(ValueError, "Duplicate"):
            replace_turn_rows([old_turn], replacement * 2, "train")

    def test_admission_requires_motion_during_turn(self):
        event = dict(
            id="fixture",
            admission_reasons=[],
            pad_frames=0,
            real_frames=100,
            metrics=dict(
                body_turn_rad=1.0,
                path_turn_rad=0.95,
                walking_annotation_fraction=1.0,
                conflicting_action_fraction=0.0,
                path_m=1.2,
                excursion_m=0.8,
                mean_speed_m_s=0.5,
                event_path_m=0.8,
                event_excursion_m=0.5,
                moving_during_turn_fraction=0.9,
                yaw_weighted_moving_fraction=0.95,
                direction_consistency=0.9,
                forward_alignment_fraction=0.9,
                entry_speed_m_s=0.3,
                exit_speed_m_s=0.3,
                ankle_relative_excursion_m=[0.2, 0.2],
                root_height_range_m=0.08,
            ),
        )
        check_admission(event)
        # Walking before/after a stationary pivot cannot qualify by overall path alone.
        for field, value in [
            ("event_path_m", 0.01),
            ("yaw_weighted_moving_fraction", 0.1),
            ("path_turn_rad", 0.0),
        ]:
            rejected = copy.deepcopy(event)
            rejected["metrics"][field] = value
            with self.assertRaisesRegex(ValueError, "does not pass"):
                check_admission(rejected)


if __name__ == "__main__":
    unittest.main()
