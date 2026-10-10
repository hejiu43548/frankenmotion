"""Demo2 independent entry; frozen Demo1 primitives remain unchanged."""

import hydra
from g1_demo import generate


@hydra.main(version_base="1.3", config_path="../config", config_name="g1_demo2_phase")
def main(config):
    if config.phase == "generate":
        generate(config)
    elif config.phase in ["retarget", "probe"]:
        from g1_demo_physics import run_phase

        run_phase(config)
    else:
        from g1_demo2_phase_pipeline import run_phase

        run_phase(config)


if __name__ == "__main__":
    main()
