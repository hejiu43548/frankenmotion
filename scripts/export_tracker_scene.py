"""Export the exact mjlab G1 scene used by reference FK and native evaluation.

The bootstrap file supplies shapes to upstream environment construction only.
It is never included in the generated motion corpus or used as policy data.
"""

import hydra
from omegaconf import DictConfig
import json
from pathlib import Path
import mujoco
import torch
import mjlab.tasks
from mjlab.tasks.registry import load_env_cfg
from mjlab.envs import ManagerBasedRlEnv


@hydra.main(config_path="../config/tracker_rl", config_name="scene", version_base="1.3")
def main(arguments: DictConfig):
    torch.set_num_threads(2)
    root = Path(arguments.output)
    root.mkdir(parents=True, exist_ok=False)
    configuration = load_env_cfg("Mjlab-Tracking-Flat-Unitree-G1", play=True)
    configuration.scene.num_envs = 1
    configuration.commands["motion"].motion_file = arguments.bootstrap_motion
    configuration.events = {}
    configuration.sim.nconmax = 128
    configuration.sim.njmax = 1024
    environment = ManagerBasedRlEnv(cfg=configuration, device=arguments.device)
    robot = environment.scene["robot"]
    mujoco.mj_saveModel(environment.sim.mj_model, str(root / "scene.mjb"))
    contract = {
        "joint_names": list(robot.joint_names),
        "body_names": list(robot.body_names),
        "joint_qpos_addresses": robot.indexing.joint_q_adr.tolist(),
        "joint_velocity_addresses": robot.indexing.joint_v_adr.tolist(),
        "body_ids": robot.indexing.body_ids.tolist(),
        "default_joint_pos": robot.data.default_joint_pos[0].cpu().tolist(),
        "dt": environment.step_dt,
    }
    (root / "contract.json").write_text(json.dumps(contract, indent=2))
    print("SCENE_EXPORTED", flush=True)
    environment.close()


if __name__ == "__main__":
    main()
