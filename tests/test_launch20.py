import copy
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from omegaconf import OmegaConf

from scripts.prepare_amass20 import event_windows
from scripts.run_stages import main as supervise
from shared_motion.training.model import build_model, file_sha256
from shared_motion.training.runner import checkpoint_compatible, save_json
from tests.test_staged_training import composed
from shared_motion.training.catalog import TASK_NAMES


class Launch20Test(unittest.TestCase):
    def test_full_timeline_and_late_timed_event(self):
        annotation = dict(start=0, end=20, annotations=[])
        windows = event_windows(annotation, r"clap", 400)
        self.assertEqual(windows[0][0], 0)
        self.assertEqual(windows[-1][1], 400)
        self.assertEqual(sum(stop - start for start, stop, _ in windows), 400)
        self.assertTrue(all(40 <= stop - start <= 120 for start, stop, _ in windows))
        annotation["annotations"] = [
            dict(bodypart="action", text="clapping", start=12, end=17)
        ]
        windows = event_windows(annotation, r"clap", 400)
        self.assertEqual([(start, stop) for start, stop, _ in windows], [(240, 340)])

    def test_external_root_scope(self):
        config = composed(2)
        config.backbone = dict(_target_="tests.training_fixtures.toy_bundle")
        model = build_model(config, "task")
        imported = dict(
            format="verified_external_root_v1",
            controller_kind="charlie_root",
            backbone_sha256=model.backbone_sha256,
            backbone_config=model.backbone_configuration,
            source=dict(checkpoint_sha256="source-test-hash"),
            import_checks=dict(parameters_exact=True, forward_bitwise_equal=True),
        )
        checkpoint_compatible(imported, model, TASK_NAMES, 1)
        with self.assertRaisesRegex(ValueError, "External root import"):
            checkpoint_compatible(imported, model, TASK_NAMES, 2)
        wrong = copy.deepcopy(imported)
        wrong["import_checks"]["forward_bitwise_equal"] = False
        with self.assertRaisesRegex(ValueError, "External root import"):
            checkpoint_compatible(wrong, model, TASK_NAMES, 1)

    def test_supervisor_selects_best_and_stops_on_failure(self):
        for fail in [False, True]:
            with tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                config = OmegaConf.create(
                    dict(
                        output=str(root),
                        resume=False,
                        initial_root=str(root / "imported.pt"),
                        controller="with_root",
                        experiment="test",
                        run_date="20261009",
                        backbone="official",
                        train_manifest="train.json",
                        val_manifest="val.json",
                        cache_root=".",
                        skeleton="skeleton.npz",
                        stage2_overrides=[],
                        stage3_overrides=[],
                    )
                )
                commands = []

                class Child:
                    pid = 123456789

                    def __init__(self, command, **kwargs):
                        commands.append(command)
                        stage = len(commands) + 1
                        report = root / "reports" / f"stage{stage}"
                        artifacts = root / f"stage{stage}"
                        artifacts.mkdir(parents=True)
                        selected = artifacts / "checkpoint_000001.pt"
                        selected.write_bytes(b"best checkpoint")
                        save_json(report / "status.json", dict(state="complete"))
                        save_json(
                            report / "completion_audit.json",
                            dict(frozen_parameters_verified=True),
                        )
                        save_json(
                            report / "best.json",
                            dict(
                                step=1,
                                metric=0.1,
                                checkpoint=str(selected),
                                sha256=file_sha256(selected),
                            ),
                        )

                    def wait(self):
                        return 1 if fail else 0

                with patch("torch.cuda.is_available", return_value=True), patch(
                    "scripts.run_stages.subprocess.Popen", Child
                ):
                    if fail:
                        with self.assertRaisesRegex(RuntimeError, "Stage2 exited"):
                            supervise.__wrapped__(config)
                        self.assertEqual(len(commands), 1)
                    else:
                        supervise.__wrapped__(config)
                        self.assertEqual(len(commands), 2)
                        self.assertIn(
                            f"initial={root}/stage2/checkpoint_000001.pt", commands[1]
                        )


if __name__ == "__main__":
    unittest.main()
