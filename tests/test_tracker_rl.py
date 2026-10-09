"""Numerical and episode-boundary tests for the real tracking environment.

Run with TRACKER_RL_ARTIFACTS pointing at the prepared corpus. These tests use
actual MuJoCo assets; they deliberately do not substitute synthetic training data.
"""

import dataclasses
import json
import os
from pathlib import Path
import unittest

import mujoco
import numpy as np
from omegaconf import OmegaConf
import torch

from shared_motion.rl.environment import build_configuration
from shared_motion.rl.environment import TrackingEnvironment
from shared_motion.rl.motion import reference_preview


@unittest.skipUnless(
    os.environ.get("TRACKER_RL_ARTIFACTS"), "Prepared real corpus required"
)
class TrackerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(2)
        cls.root = Path(os.environ["TRACKER_RL_ARTIFACTS"])
        cls.configuration = OmegaConf.create(
            {
                "artifacts": str(cls.root),
                "seed": 72001,
                "num_envs": 40,
                "start_probability": 0.35,
                "preview_offsets": [],
                "save_interval": 500,
            }
        )
        environment_configuration, _ = build_configuration(
            cls.configuration, "val", evaluation=True
        )
        cls.environment = TrackingEnvironment(
            cfg=environment_configuration, device="cuda:0"
        )

    @classmethod
    def tearDownClass(cls):
        cls.environment.close()

    def test_reset_targets_are_fresh_without_advancing_time(self):
        self.environment.reset()
        command = self.environment.command_manager.get_term("motion")
        self.assertTrue(
            torch.equal(command.time_steps, command.starts[command.clip_ids])
        )
        self.assertLess(
            float((command.body_pos_relative_w - command.robot_body_pos_w).abs().max()),
            5e-3,
        )
        self.assertFalse(bool(self.environment.termination_manager.compute().any()))

    def test_reference_end_is_reported_to_ppo(self):
        self.environment.reset()
        command = self.environment.command_manager.get_term("motion")
        command.time_steps[:] = command.ends[command.clip_ids] - 1
        _, _, terminated, truncated, _ = self.environment.step(
            torch.zeros(40, 29, device="cuda:0")
        )
        self.assertTrue(bool((terminated | truncated).all()))
        self.assertTrue(
            bool(torch.all(command.time_steps < command.ends[command.clip_ids]))
        )

    def test_preview_clamps_within_each_clip(self):
        self.environment.reset()
        command = self.environment.command_manager.get_term("motion")
        preview = reference_preview(self.environment, [100000])
        expected = command.motion.joint_pos[command.ends[command.clip_ids] - 1]
        torch.testing.assert_close(preview[:, :29], expected)
        self.assertTrue(bool(torch.isfinite(preview).all()))

    def test_preview_does_not_change_rewards(self):
        baseline, _ = build_configuration(self.configuration)
        preview_configuration = OmegaConf.merge(
            self.configuration, {"preview_offsets": [5, 10, 20]}
        )
        preview, _ = build_configuration(preview_configuration)
        self.assertEqual(
            dataclasses.asdict(baseline)["rewards"],
            dataclasses.asdict(preview)["rewards"],
        )
        self.assertEqual(len(baseline.rewards), 9)

    def test_splits_have_no_shared_seeds_or_files(self):
        records = {
            split: json.loads(
                (self.root / "motion" / split / "clips.json").read_text()
            )["records"]
            for split in ["train", "val", "test"]
        }
        for first, second in [("train", "val"), ("train", "test"), ("val", "test")]:
            self.assertFalse(
                {row["seed"] for row in records[first]}
                & {row["seed"] for row in records[second]}
            )
            self.assertFalse(
                {row["sha256"] for row in records[first]}
                & {row["sha256"] for row in records[second]}
            )
        self.assertTrue(
            all(len({row["task"] for row in rows}) == 20 for rows in records.values())
        )

    def test_body_origin_velocity_matches_finite_difference(self):
        model = mujoco.MjModel.from_binary_path(str(self.root / "scene/scene.mjb"))
        data = mujoco.MjData(model)
        data.qvel[:] = np.linspace(-0.5, 0.5, model.nv)
        mujoco.mj_forward(model, data)
        body_id = model.body("robot/left_wrist_yaw_link").id
        before = data.xpos[body_id].copy()
        velocity = np.empty(6)
        mujoco.mj_objectVelocity(
            model, data, mujoco.mjtObj.mjOBJ_XBODY, body_id, velocity, 0
        )
        mujoco.mj_integratePos(model, data.qpos, data.qvel, 1e-6)
        mujoco.mj_forward(model, data)
        np.testing.assert_allclose(
            (data.xpos[body_id] - before) / 1e-6, velocity[3:], atol=1e-6
        )


if __name__ == "__main__":
    unittest.main()
