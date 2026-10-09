"""Fixed upstream BeyondMimic reward recipe; no task-specific rewards."""

import dataclasses
from pathlib import Path

import mjlab.tasks
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import ObservationTermCfg
from mjlab.managers import TerminationTermCfg
from mjlab.tasks.registry import load_env_cfg
from mjlab.tasks.registry import load_rl_cfg

from .motion import MultiClipCommandCfg
from .motion import reference_finished
from .motion import reference_preview


def build_configuration(configuration, split="train", evaluation=False):
    environment = load_env_cfg("Mjlab-Tracking-Flat-Unitree-G1", play=False)
    environment.seed = configuration.seed
    environment.scene.num_envs = configuration.num_envs
    environment.sim.nconmax = 128
    environment.sim.njmax = 1024
    environment.episode_length_s = 10.0
    original = environment.commands["motion"]
    command_fields = {
        field.name: getattr(original, field.name)
        for field in dataclasses.fields(original)
    }
    directory = Path(configuration.artifacts) / "motion" / split
    command_fields.update(
        motion_file=str(directory / "motions.npz"),
        sampling_mode="uniform",
        metadata=str(directory / "clips.json"),
        evaluation=evaluation,
        start_probability=configuration.start_probability,
    )
    environment.commands["motion"] = MultiClipCommandCfg(**command_fields)
    environment.terminations["reference_end"] = TerminationTermCfg(
        func=reference_finished, time_out=False
    )
    if configuration.preview_offsets:
        for group in ["actor", "critic"]:
            environment.observations[group].terms["reference_preview"] = (
                ObservationTermCfg(
                    func=reference_preview,
                    params={"offsets": list(configuration.preview_offsets)},
                )
            )
    if evaluation:
        environment.events = {}
        environment.observations["actor"].enable_corruption = False
        command = environment.commands["motion"]
        command.pose_range = {}
        command.velocity_range = {}
        command.joint_position_range = (0.0, 0.0)
    agent = load_rl_cfg("Mjlab-Tracking-Flat-Unitree-G1")
    agent.logger = "tensorboard"
    agent.save_interval = configuration.save_interval
    return environment, agent


class TrackingEnvironment(ManagerBasedRlEnv):
    """Ensure initial reward and termination targets refer to the reset state."""

    def reset(self, **kwargs):
        observations, extras = super().reset(**kwargs)
        self.command_manager.get_term("motion").refresh_reference_frame()
        return observations, extras
