"""Development experiment: adapt an official dance tracker to ONE backward reference.
Not a universal controller, not an independent confirmation experiment.
"""
import os,json,time,argparse,types
import numpy as np
from pathlib import Path
os.environ['WANDB_MODE']='disabled';os.environ['MUJOCO_GL']='egl'
import torch
from dataclasses import asdict
import mjlab.tasks
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
ROOT=Path('/home/pku/frankenmotion/outputs_amass/franken_improve_20261003')
a=argparse.ArgumentParser();a.add_argument('--task',default='back_walk');a.add_argument('--steps',type=int,default=1200);a.add_argument('--envs',type=int,default=128);a.add_argument('--root-weight',type=float,default=.5);a.add_argument('--all-commands',action='store_true');a.add_argument('--root-std',type=float,default=.3);args=a.parse_args()
folder=ROOT/('beyondmimic_finetune_'+args.task+('' if args.root_weight==.5 else '_root'+str(args.root_weight))+('_all' if args.all_commands else '')+('' if args.root_std==.3 else '_std'+str(args.root_std)));folder.mkdir(exist_ok=True)
torch.set_num_threads(2);torch.manual_seed(5103);torch.cuda.set_per_process_memory_fraction(.12)
name='Mjlab-Tracking-Flat-Unitree-G1';cfg=load_env_cfg(name,play=False);cfg.scene.num_envs=args.envs;cfg.seed=5103;cfg.commands['motion'].motion_file=str(ROOT/'beyondmimic_development'/(args.task+'_p0_s0_c2_motion.npz'))
cfg.rewards['motion_global_root_pos'].weight=args.root_weight;cfg.rewards['motion_global_root_pos'].params['std']=args.root_std
boundaries=None
if args.all_commands:
 source_tasks=['back_walk','sidestep','walk'] if args.task=='locomotion' else [args.task];clips=[np.load(ROOT/'beyondmimic_development'/(task+f'_p0_s0_c{i}_motion.npz')) for task in source_tasks for i in range(5)];boundaries=np.cumsum([len(x['joint_pos']) for x in clips]);combined=folder/'training_motions.npz';np.savez_compressed(combined,fps=50.,**{k:np.concatenate([x[k] for x in clips]) for k in clips[0].files if k!='fps'});cfg.commands['motion'].motion_file=str(combined)
agent=load_rl_cfg(name);agent.logger='tensorboard';agent.save_interval=200;agent.algorithm.learning_rate=1e-4;agent.algorithm.schedule='fixed';agent.algorithm.entropy_coef=.001
(folder/'protocol.json').write_text(json.dumps(dict(task=args.task,root_position_reward_std=args.root_std,all_commands=args.all_commands,clip_end_frames=boundaries.tolist() if boundaries is not None else None,clip_reset='standard reference-state resampling at every source clip end; never track concatenation jumps',root_position_reward_weight=args.root_weight,command_index=2,training_reference=cfg.commands['motion'].motion_file,iterations=args.steps,num_envs=args.envs,initial_checkpoint='/home/pku/frankenmotion/work/beyondmimic_demo.pt',actor_and_critic_restored=True,optimizer_restored=False,learning_rate=1e-4,evaluation='all five development commands, shared source; no independent generalization claim',runner=asdict(agent)),indent=2))
e=ManagerBasedRlEnv(cfg=cfg,device='cuda:0')
if boundaries is not None:
 term=e.command_manager.get_term('motion');ends=torch.as_tensor(boundaries,device='cuda:0');original=term._update_command
 def segmented_update(self):
  index=torch.bucketize(self.time_steps,ends,right=True).clamp_max(len(ends)-1);cross=self.time_steps+1>=ends[index];ids=torch.where(cross)[0]
  if ids.numel():
   self._resample_command(ids);index=torch.bucketize(self.time_steps[ids],ends,right=True).clamp_max(len(ends)-1);self.time_steps[ids]=torch.minimum(self.time_steps[ids],ends[index]-2)
  original()
 term._update_command=types.MethodType(segmented_update,term)
env=RslRlVecEnvWrapper(e,clip_actions=agent.clip_actions);runner=MjlabOnPolicyRunner(env,asdict(agent),str(folder),device='cuda:0');runner.load('/home/pku/frankenmotion/work/beyondmimic_demo.pt',load_cfg={'actor':True,'critic':True},strict=True,map_location='cuda:0')
start=time.monotonic();runner.learn(num_learning_iterations=args.steps,init_at_random_ep_len=True);(folder/'complete.json').write_text(json.dumps(dict(iterations=args.steps,wall_s=time.monotonic()-start,max_torch_memory_mb=torch.cuda.max_memory_allocated()/2**20)));env.close()
