"""Headless own-reference gate for the official mjlab BeyondMimic demo."""
import os,json,time
os.environ['WANDB_MODE']='disabled';os.environ.setdefault('MUJOCO_GL','egl')
from pathlib import Path
from dataclasses import asdict
import torch
import numpy as np
import mjlab.tasks
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg,load_runner_cls
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
ROOT=Path('/home/pku/frankenmotion/outputs_amass/franken_improve_20261003');OUT=ROOT/'beyondmimic_gate';OUT.mkdir(exist_ok=True)
torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.12)
name='Mjlab-Tracking-Flat-Unitree-G1';cfg=load_env_cfg(name,play=True);cfg.scene.num_envs=1;cfg.commands['motion'].motion_file='/home/pku/frankenmotion/work/beyondmimic_demo_motion.npz'
cfg.commands['motion'].sampling_mode='start';print('Creating environment',flush=True)
env0=ManagerBasedRlEnv(cfg=cfg,device='cuda:0');agent=load_rl_cfg(name);env=RslRlVecEnvWrapper(env0,clip_actions=agent.clip_actions);runner=(load_runner_cls(name) or MjlabOnPolicyRunner)(env,asdict(agent),device='cuda:0')
runner.load('/home/pku/frankenmotion/work/beyondmimic_demo.pt',load_cfg={'actor':True},strict=True,map_location='cuda:0');policy=runner.get_inference_policy(device='cuda:0');obs,_=env.reset();states=[];rewards=[];dones=[];errors=[];start=time.monotonic()
with torch.inference_mode():
 for i in range(1000):
  act=policy(obs);obs,rew,done,info=env.step(act);robot=env0.scene['robot'];states.append(robot.data.root_link_pos_w.detach().cpu().numpy().copy());rewards.append(float(rew.mean()));dones.append(bool(done.any())); motion=env0.command_manager.get_term('motion');errors.append({k:float(v.mean()) for k,v in motion.metrics.items()})
  if i%100==0:print(i,'rewards',np.mean(rewards[-100:]),'resets',sum(dones),flush=True)
r=dict(steps=1000,resets=sum(dones),reward_mean=float(np.mean(rewards)),wall_s=time.monotonic()-start,max_torch_memory_mb=torch.cuda.max_memory_allocated()/2**20,source='Official dance reference; not the FrankenMotion benchmark', tracking_errors={k:float(np.mean([e[k] for e in errors])) for k in errors[0]}, reset_steps=np.flatnonzero(dones).tolist())
(OUT/'result.json').write_text(json.dumps(r,indent=2));np.savez_compressed(OUT/'trajectory.npz',root_position=np.array(states),reward=rewards,done=dones);print(r,flush=True);env.close()
