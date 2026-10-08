import sys
sys.path.insert(0,"/home/pku/frankenmotion/work")
"""Joint PPO: one actor, critic and action interface for all motion clips."""
import os,json,time,argparse,hashlib,dataclasses,shutil
from pathlib import Path
os.environ['WANDB_MODE']='disabled';os.environ['MUJOCO_GL']='egl'
import torch
import mjlab.tasks
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import TerminationTermCfg,RewardTermCfg
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
from unified_motion_20261004 import reference_end
from table_training_20261005 import table_scene
from joint_sampling import BroadMotionCfg as MultiMotionCfg
from shared_rewards import pelvis_position_dense,pelvis_orientation,distal_position,reference_velocity_dense,bilateral_foot_height_dense
R=Path('/home/pku/frankenmotion');U=R/'outputs_amass/universal_tracker_20261006'
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--dataset',default='development');p.add_argument('--preview',action='store_true');p.add_argument('--steps',type=int,default=3000);p.add_argument('--envs',type=int,default=128);p.add_argument('--seed',type=int,default=4104);p.add_argument('--initial',default=str(R/'work/beyondmimic_demo.pt'));p.add_argument('--root-weight',type=float,default=2.);p.add_argument('--root-std',type=float,default=.15);p.add_argument('--body-pos-std',type=float,default=.3);p.add_argument('--body-pos-weight',type=float,default=1.);p.add_argument('--joint-weight',type=float,default=0.);p.add_argument('--joint-std',type=float,default=.3);p.add_argument('--action-rate-weight',type=float,default=-.1);p.add_argument('--learning-rate',type=float,default=1e-4);p.add_argument('--task-balanced-slots',action='store_true');p.add_argument('--calibrate-preview',action='store_true');p.add_argument('--joint-worst-count',type=int,default=0);p.add_argument('--entropy-coef',type=float,default=.001);p.add_argument('--episode-seconds',type=float,default=10.);p.add_argument('--root-ori-weight',type=float,default=.5);p.add_argument('--root-ori-std',type=float,default=.4);p.add_argument('--root-wide-weight',type=float,default=0.);p.add_argument('--root-wide-std',type=float,default=.5);p.add_argument('--long-preview',action='store_true');p.add_argument('--reset-training-rng',action='store_true');p.add_argument('--extended-rewards',action='store_true');p.add_argument('--table-fraction',type=float,default=.15);p.add_argument('--retention-weight',type=float,default=0.);p.add_argument('--physical-smoothness',action='store_true');p.add_argument('--velocity-weight',type=float,default=.5);p.add_argument('--pelvis-relative-weight',type=float,default=0.);p.add_argument('--new-motion-fraction',type=float,default=.25);args=p.parse_args();assert not args.long_preview or args.preview;assert not args.calibrate_preview or args.preview
folder=U/'training'/args.name;folder.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.manual_seed(args.seed);torch.cuda.set_per_process_memory_fraction(.14)
cfg=load_env_cfg('Mjlab-Tracking-Flat-Unitree-G1',play=False);cfg.scene.num_envs=args.envs;cfg.episode_length_s=args.episode_seconds;cfg.seed=args.seed;cfg.sim.nconmax=256;cfg.sim.njmax=2048;cfg.scene.spec_fn=table_scene
if args.preview:
 from unified_preview_20261004 import configure_preview,OFFSETS,LONG_OFFSETS
 configure_preview(cfg,LONG_OFFSETS if args.long_preview else OFFSETS)
old=cfg.commands['motion'];kw={f.name:getattr(old,f.name) for f in dataclasses.fields(old)};kw.update(motion_file=str(U/args.dataset/'training_motions.npz'),sampling_mode='uniform');cfg.commands['motion']=MultiMotionCfg(**kw,clip_metadata=str(U/args.dataset/'clips.json'),task_balanced_slots=args.task_balanced_slots,table_fraction=args.table_fraction,new_motion_fraction=args.new_motion_fraction)
# Finite reference clips are genuine terminal episodes, never silent simulator resets.
cfg.terminations['reference_end']=TerminationTermCfg(func=reference_end,time_out=False)
cfg.rewards['motion_global_root_pos'].weight=args.root_weight;cfg.rewards['motion_global_root_pos'].params['std']=args.root_std
cfg.rewards['motion_global_root_ori'].weight=args.root_ori_weight;cfg.rewards['motion_global_root_ori'].params['std']=args.root_ori_std
if args.root_wide_weight:
 cfg.rewards['motion_global_root_pos_wide']=dataclasses.replace(cfg.rewards['motion_global_root_pos'],weight=args.root_wide_weight,params={'command_name':'motion','std':args.root_wide_std})
cfg.rewards['motion_body_pos'].params['std']=args.body_pos_std;cfg.rewards['motion_body_pos'].weight=args.body_pos_weight;cfg.rewards['action_rate_l2'].weight=args.action_rate_weight
if args.joint_weight:
 from unified_rewards_20261004 import joint_reference_accuracy
 cfg.rewards['joint_reference_accuracy']=RewardTermCfg(func=joint_reference_accuracy,weight=args.joint_weight,params={'command_name':'motion','std':args.joint_std,'worst_count':args.joint_worst_count})
if args.extended_rewards:
 for key,func,weight in [('pelvis_position_dense',pelvis_position_dense,2.),('pelvis_orientation',pelvis_orientation,2.),('distal_position',distal_position,2.),('reference_velocity_dense',reference_velocity_dense,args.velocity_weight),('bilateral_foot_height_dense',bilateral_foot_height_dense,2.)]:
  cfg.rewards[key]=RewardTermCfg(func=func,weight=weight)
if args.pelvis_relative_weight:
 from adaptive_rewards import pelvis_relative_distal_position
 cfg.rewards['pelvis_relative_distal_position']=RewardTermCfg(func=pelvis_relative_distal_position,weight=args.pelvis_relative_weight)
if args.physical_smoothness:
 from adaptive_rewards import reference_relative_target_rate
 cfg.rewards['action_rate_l2']=RewardTermCfg(func=reference_relative_target_rate,weight=args.action_rate_weight)
agent=load_rl_cfg('Mjlab-Tracking-Flat-Unitree-G1');agent.logger='tensorboard';agent.save_interval=500;agent.algorithm.learning_rate=args.learning_rate;agent.algorithm.schedule='fixed';agent.algorithm.entropy_coef=args.entropy_coef
protocol=dict(preview_offsets=list(LONG_OFFSETS if args.long_preview else OFFSETS) if args.preview else [],arguments=vars(args),actor='single shared MLP; no task id, routing or mixture of experts',conditioning='reference joint positions/velocities, anchor error, proprioception; shared future frames specified in preview_offsets' if args.preview else 'reference joint positions/velocities, anchor error, proprioception; unchanged BM observation schema',sampling=('fixed balanced environment slots per data task; uniform clips within task' if args.task_balanced_slots else 'uniform over every corpus clip')+'; 35% start, otherwise uniform phase; task metadata never enters actor observations',boundary='finite-horizon terminal episode via termination manager; no silent resets',contact_capacity=256,constraint_capacity=2048,development_manifest_sha256=hashlib.sha256((U/args.dataset/'manifest.json').read_bytes()).hexdigest(),initial_sha256=hashlib.sha256(Path(args.initial).read_bytes()).hexdigest(),runner=dataclasses.asdict(agent))
snapshot=folder/'source_snapshot';snapshot.mkdir();sources={}
for filename in ['train','motion','preview','rewards','calibration']:
 source=R/'work'/f'unified_{filename}_20261004.py'
 if source.exists():
  shutil.copy2(source,snapshot/source.name);sources[source.name]=hashlib.sha256(source.read_bytes()).hexdigest()
for source in [Path(__file__),Path(__file__).parent/'shared_rewards.py',Path(__file__).parent/'retention.py',Path(__file__).parent/'adaptive_rewards.py',Path(__file__).parent/'joint_sampling.py',R/'work/table_training_20261005.py']:
 shutil.copy2(source,snapshot/source.name);sources[source.name]=hashlib.sha256(source.read_bytes()).hexdigest()
protocol['training_source_sha256']=sources
protocol['reward_scope']='Same global pelvis, distal-body and velocity formulas for all motion classes; no semantic task dispatch.'
(folder/'protocol.json').write_text(json.dumps(protocol,indent=2));e=ManagerBasedRlEnv(cfg=cfg,device='cuda:0');e.sim.expand_model_fields(('body_pos','body_quat'));e.table_fields_ready=True;env=RslRlVecEnvWrapper(e,clip_actions=agent.clip_actions);runner=MjlabOnPolicyRunner(env,dataclasses.asdict(agent),str(folder),device='cuda:0');runner.load(args.initial,load_cfg={'actor':True,'critic':True},strict=True,map_location='cuda:0')
from retention import attach_retention
if args.retention_weight>0:retention_handle,retention_stats=attach_retention(runner,folder,weight=args.retention_weight)
original_save=runner.save
def save_verified(path,infos=None):
 path=Path(path);temporary=path.with_suffix(path.suffix+'.tmp');original_save(str(temporary),infos);os.replace(temporary,path);path.with_suffix(path.suffix+'.ready.json').write_text(json.dumps({'checkpoint_sha256':hashlib.sha256(path.read_bytes()).hexdigest(),'bytes':path.stat().st_size}))
runner.save=save_verified
if args.calibrate_preview:
 from unified_calibration_20261004 import calibrate_preview
 torch.manual_seed(args.seed+77000) if args.reset_training_rng else None
 (folder/'preview_calibration.json').write_text(json.dumps(calibrate_preview(env,runner,preview_dim=(335 if args.long_preview else 201)),indent=2))
if args.reset_training_rng:torch.manual_seed(args.seed)
# Deliberately reach every clip end and verify PPO sees done before a new clip.
env.reset();term=e.command_manager.get_term('motion');before=term.sample_counts.sum().item();term.time_steps[:]=term.ends[term.clip_ids]-1
_,_,done,_=env.step(torch.zeros((args.envs,29),device='cuda:0'))
assert bool(done.all()) and term.sample_counts.sum().item()>=before+args.envs
assert bool(torch.all(term.time_steps<term.ends[term.clip_ids]))
(folder/'boundary_audit.json').write_text(json.dumps(dict(forced_terminal_envs=args.envs,done_count=int(done.sum()),all_boundaries_reported=True,no_cross_clip_bootstrap=True)))
env.reset()
start=time.monotonic();runner.learn(num_learning_iterations=args.steps,init_at_random_ep_len=True);term=e.command_manager.get_term('motion');(folder/'complete.json').write_text(json.dumps(dict(iterations=args.steps,wall_s=time.monotonic()-start,clip_sample_counts=term.sample_counts.tolist(),command_step_counts=term.command_step_counts.tolist(),task_names=term.task_names,task_command_steps=torch.bincount(term.clip_task,weights=term.command_step_counts.float(),minlength=len(term.task_names)).tolist(),max_torch_memory_mb=torch.cuda.max_memory_allocated()/2**20)));env.close()
