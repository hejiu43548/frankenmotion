"""Selection must preserve the declared ranking and reject test leakage."""

import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from scripts.select_tracker_checkpoint import select_checkpoint


class SelectionTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.policy = self.root / "policy.pt"
        self.scene = self.root / "scene.mjb"
        self.policy.write_bytes(b"selection-only synthetic policy")
        self.scene.write_bytes(b"selection-only synthetic scene")
        self.policy_hash = hashlib.sha256(self.policy.read_bytes()).hexdigest()
        self.scene_hash = hashlib.sha256(self.scene.read_bytes()).hexdigest()
        self.contract = self.root / "contract.json"
        self.contract.write_text(
            json.dumps(
                {
                    "schema_version": 2,
                    "policy_sha256": self.policy_hash,
                    "checkpoint_sha256": "synthetic-checkpoint",
                }
            )
        )

    def configuration(self, scores):
        candidates = []
        for iteration, success, complete, error in scores:
            metrics = {
                "configuration": {
                    "split": "val",
                    "perturbation": 0,
                    "seed": 61001,
                    "policy": str(self.policy),
                    "scene": str(self.scene),
                    "contract": str(self.contract),
                },
                "backend": "native MuJoCo synthetic selection fixture",
                "references": {"walk:42001": "synthetic-reference"},
                "scene_sha256": self.scene_hash,
                "policy_sha256": self.policy_hash,
                "episodes": [
                    {"task": "walk", "motion_seed": 42001, "expected_frames": 298}
                ],
                "macro": {
                    "tracking_success": success,
                    "complete": complete,
                    "root_m_failure_penalized": error,
                },
            }
            path = self.root / f"metrics_{iteration}.json"
            path.write_text(json.dumps(metrics))
            candidates.append({"iteration": iteration, "metrics": str(path)})
        return {
            "experiment": "synthetic_selection_test",
            "expected_iterations": [entry["iteration"] for entry in candidates],
            "expected_episodes": 1,
            "candidates": candidates,
        }

    def test_ranking_and_ties(self):
        configuration = self.configuration(
            [
                (40, 0.8, 1.0, 0.1),
                (30, 0.8, 1.0, 0.1),
                (20, 0.8, 1.0, 0.2),
                (10, 0.8, 0.9, 0.05),
                (0, 0.7, 1.0, 0.01),
            ]
        )
        result = select_checkpoint(configuration)
        self.assertEqual(
            [entry["iteration"] for entry in result["ranking"]], [30, 40, 20, 10, 0]
        )

    def test_incomplete_set_rejected(self):
        configuration = self.configuration([(10, 0.8, 1.0, 0.1)])
        configuration["expected_iterations"].append(20)
        with self.assertRaisesRegex(ValueError, "Candidate set"):
            select_checkpoint(configuration)

    def test_test_split_rejected(self):
        configuration = self.configuration([(10, 0.8, 1.0, 0.1)])
        path = Path(configuration["candidates"][0]["metrics"])
        metrics = json.loads(path.read_text())
        metrics["configuration"]["split"] = "test"
        path.write_text(json.dumps(metrics))
        with self.assertRaisesRegex(ValueError, "validation"):
            select_checkpoint(configuration)

    def test_changed_policy_rejected(self):
        configuration = self.configuration([(10, 0.8, 1.0, 0.1)])
        self.policy.write_bytes(b"changed")
        with self.assertRaisesRegex(ValueError, "Policy changed"):
            select_checkpoint(configuration)


if __name__ == "__main__":
    unittest.main()
