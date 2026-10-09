"""Select the physical action mapping without changing PPO's sampled action."""

from mjlab.rl import RslRlVecEnvWrapper

from .residual import ReferenceResidualWrapper


def make_wrapper(environment, configuration, clip_actions=None):
    if configuration.get("sonic_directory", None):
        if configuration.get("reference_residual", False):
            raise ValueError("Cannot combine kinematic and SONIC residual mappings")
        if list(configuration.preview_offsets) != [5, 10, 20]:
            raise ValueError(
                "SONIC residual deployment currently requires preview [5,10,20]"
            )
        from .sonic_residual import SonicResidualWrapper

        return SonicResidualWrapper(
            environment,
            configuration.sonic_directory,
            configuration.sonic_contract,
            clip_actions=clip_actions,
        )
    wrapper = (
        ReferenceResidualWrapper
        if configuration.get("reference_residual", False)
        else RslRlVecEnvWrapper
    )
    return wrapper(environment, clip_actions=clip_actions)
