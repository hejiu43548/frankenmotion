import json
from pathlib import Path
import tempfile
import unittest

import torch

from shared_motion.adapter.catalog import NEW, quantities
from shared_motion.adapter.kinematics import quantity
from shared_motion.training.catalog import HUMAN_HEIGHT, TASK_NAMES, measure
from shared_motion.training.geometry import Skeleton
from shared_motion.training.model import file_sha256
from shared_motion.training.runner import validate_source_migration
from shared_motion.training.turn import signed_turn
from tests.training_fixtures import make_data


def reference_measure(skeleton, motion, task_indices, lengths):
    values = []
    for sample_index, task_index in enumerate(task_indices.tolist()):
        name = TASK_NAMES[task_index]
        positions = skeleton(
            motion[sample_index : sample_index + 1, : int(lengths[sample_index])]
        )
        if name == "turn":
            result = signed_turn(positions)
        elif name == "march":
            # Staged march now means stationary running; legacy release metrics
            # keep their original first-frame standing baseline.
            feet = positions[:, :, [7, 8], 2]
            result = (feet.amax(1) - torch.quantile(feet, 0.05, dim=1)).mean(1)
        elif name in NEW:
            result = quantities(positions)[name]
        else:
            result = quantity(positions, name, scale=HUMAN_HEIGHT / skeleton.height)
        values.append(result[0])
    return torch.stack(values)


class BatchedMeasurementTest(unittest.TestCase):
    def test_values_gradients_variable_lengths_and_nan_padding(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as directory:
            data = make_data(Path(directory) / "data")
            skeleton = Skeleton(data / "skeleton.npz").double()
            generator = torch.Generator().manual_seed(411)
            motion = torch.randn(40, 120, 205, generator=generator, dtype=torch.float64)
            motion[..., 3] *= 0.2
            tasks = torch.arange(40) % 20
            lengths = 64 + torch.arange(40) % 57
            for sample_index, frames in enumerate(lengths.tolist()):
                motion[sample_index, frames:] = float("nan")
            original = motion.clone().requires_grad_()
            batched = motion.clone().requires_grad_()
            expected = reference_measure(skeleton, original, tasks, lengths)
            calls = []
            handle = skeleton.register_forward_hook(
                lambda module, inputs, output: calls.append(1)
            )
            actual = measure(skeleton, batched, tasks, lengths)
            handle.remove()
            self.assertEqual(len(calls), 1)
            torch.testing.assert_close(actual, expected, rtol=1e-10, atol=1e-10)
            weights = torch.linspace(0.1, 2, 40, dtype=torch.float64)
            (expected * weights).sum().backward()
            (actual * weights).sum().backward()
            self.assertTrue(torch.isfinite(batched.grad).all())
            torch.testing.assert_close(
                batched.grad, original.grad, rtol=1e-9, atol=1e-9
            )

    def test_source_migration_rejects_recipe_changes_and_tampering(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parent = root / "checkpoint.pt"
            parent.write_bytes(b"immutable checkpoint fixture")
            old = dict(sources={"catalog.py": "old"}, schedule={"steps": 70000})
            new = dict(old, sources={"catalog.py": "new"})
            checkpoint = dict(protocol=old, step=30000)
            (root / "protocol.json").write_text(json.dumps(old))
            evidence = root / "evidence.json"
            evidence.write_text(
                json.dumps(
                    dict(
                        passed=True,
                        old_sources=old["sources"],
                        new_sources=new["sources"],
                    )
                )
            )
            migration = root / "migration.json"
            migration.write_text(
                json.dumps(
                    dict(
                        evidence=str(evidence),
                        evidence_sha256=file_sha256(evidence),
                        parent_checkpoint_sha256=file_sha256(parent),
                        parent_step=30000,
                        parent_report=str(root),
                    )
                )
            )
            self.assertEqual(
                validate_source_migration(checkpoint, new, parent, migration)[
                    "parent_step"
                ],
                30000,
            )
            with self.assertRaises(ValueError):
                validate_source_migration(
                    checkpoint, dict(new, schedule={"steps": 80000}), parent, migration
                )
            evidence.write_text("{}")
            with self.assertRaises(ValueError):
                validate_source_migration(checkpoint, new, parent, migration)
