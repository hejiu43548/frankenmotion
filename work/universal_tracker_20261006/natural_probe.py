"""Evaluate one shared tracker with a fixed observation/action contract on all requests.

The wrapper selects standing entry and full-horizon physical termination; raw
trajectories are independently audited after simulation.
"""
import os,sys,json,time
from pathlib import Path
os.environ['WANDB_MODE']='disabled';os.environ['MUJOCO_GL']='egl'
import numpy as np
import torch
from scipy.spatial.transform import Rotation,Slerp
from dataclasses import asdict
import mjlab.tasks
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg,load_runner_cls
from mjlab.envs import ManagerBasedRlEnv
from mjlab.managers import TerminationTermCfg
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
from mjlab.tasks.tracking.mdp.commands import MotionLoader
ROOT=Path('/home/pku/frankenmotion/outputs_amass/franken_improve_20261003');BASE=ROOT.parent/'franken_eleven_20261003'
sys.path[:0]=[str(Path('/home/pku/frankenmotion/work')),str(BASE/'code')]
import transfer as tr
OUT=ROOT/os.environ.get('BM_OUT','beyondmimic_development_v2');OUT.mkdir(exist_ok=True)
torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.12);torch.manual_seed(4103)
name='Mjlab-Tracking-Flat-Unitree-G1';cfg=load_env_cfg(name,play=True);cfg.scene.num_envs=1;cfg.commands['motion'].motion_file='/home/pku/frankenmotion/work/beyondmimic_demo_motion.npz';cfg.commands['motion'].joint_position_range=(0.,0.)
if os.environ.get('BM_TERMINATION')=='physical':
 def physical_fall(env):
  data=env.scene['robot'].data;q=data.root_link_quat_w;w,x,y,z=q.unbind(-1);roll=torch.atan2(2*(w*x+y*z),1-2*(x*x+y*y));pitch=torch.asin((2*(w*y-z*x)).clamp(-1,1));finite=torch.isfinite(data.root_link_pos_w).all(-1)&torch.isfinite(q).all(-1)&torch.isfinite(data.joint_pos).all(-1)
  return (data.root_link_pos_w[:,2]<.35)|((roll.square()+pitch.square()).sqrt()>np.pi/3)|(~finite)
 cfg.terminations={'physical_fall':TerminationTermCfg(func=physical_fall)}
cfg.sim.nconmax=256;cfg.sim.njmax=2048
if os.environ.get('UNIFIED_PREVIEW')=='1':
 from unified_preview_20261004 import configure_preview,checkpoint_offsets
 offsets=checkpoint_offsets(os.environ['BM_CHECKPOINT']);assert offsets;configure_preview(cfg,offsets)
e=ManagerBasedRlEnv(cfg=cfg,device='cuda:0');agent=load_rl_cfg(name);env=RslRlVecEnvWrapper(e,clip_actions=agent.clip_actions);runner=(load_runner_cls(name) or MjlabOnPolicyRunner)(env,asdict(agent),device='cuda:0');runner.load(os.environ.get('BM_CHECKPOINT','/home/pku/frankenmotion/work/beyondmimic_demo.pt'),load_cfg={'actor':True},strict=True,map_location='cuda:0');policy=runner.get_inference_policy(device='cuda:0')
robot=e.scene['robot'];motion=e.command_manager.get_term('motion');model=tr.rt.load_model();names=[model.joint(i).name for i in range(1,model.njnt)];order=[names.index(n) for n in robot.joint_names];inverse=np.argsort(order)
(OUT/'contract.json').write_text(json.dumps(dict(preview_offsets=list(offsets) if os.environ.get('UNIFIED_PREVIEW')=='1' else [],contact_capacity_nconmax=256,constraint_capacity_njmax=2048,joint_names=list(robot.joint_names),body_names=list(robot.body_names),source_joint_names=names,initialization='exact first reference pose and velocity, no joint noise; simulator play-mode randomizations otherwise preserved',policy=os.environ.get('BM_CHECKPOINT','official dance-specific checkpoint'),entry=os.environ.get('BM_ENTRY','reference'),termination=os.environ.get('BM_TERMINATION','original_tracking'),fps=50),indent=2))
rows=json.loads(Path(os.environ.get('BM_RESULTS',str(ROOT/'physical_development_results.json'))).read_text());rows=[r for r in rows if r.get('variant','uniform')=='uniform' and (not os.environ.get('BM_TASK') or r['task'] in os.environ['BM_TASK'].split(','))]

def tensor(x):return torch.as_tensor(x,dtype=torch.float32,device='cuda:0')
def convert(states,path):
 t=np.arange(len(states))*.05;tt=np.arange(int(np.ceil(t[-1]/.02))+1)*.02;sample_t=np.minimum(tt,t[-1]);pos=np.stack([np.interp(sample_t,t,c) for c in states[:,:3].T],1);rots=Slerp(t,Rotation.from_quat(states[:,[4,5,6,3]]))(sample_t);quat=rots.as_quat()[:,[3,0,1,2]]
 q=np.stack([np.interp(sample_t,t,c) for c in states[:,7:].T],1)[:,order];dq=np.gradient(q,.02,axis=0);vel=np.gradient(pos,.02,axis=0);av=(rots[1:]*rots[:-1].inv()).as_rotvec()/.02;av=np.r_[av,av[-1:]]
 if os.environ.get('BM_CPU_FK')=='1':
  from mjlab_cpu_fk_20261003 import convert_cpu
  arrays=convert_cpu(e,robot,pos,quat,q,dq,vel,av,rots,path)
  if os.environ.get('BM_FK_PARITY_DIR'):
   previous=np.load(Path(os.environ['BM_FK_PARITY_DIR'])/path.name);errs={k:float(np.max(abs(arrays[k]-previous[k]))) for k in arrays};assert max(errs.values())<3e-4,errs;(OUT/(path.stem+'_parity.json')).write_text(json.dumps(errs))
  return len(tt)
 log={k:[] for k in ['joint_pos','joint_vel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w']}
 for i in range(len(tt)):
  root=np.r_[pos[i],quat[i],vel[i],av[i]][None];robot.write_root_state_to_sim(tensor(root));robot.write_joint_state_to_sim(tensor(q[i:i+1]),tensor(dq[i:i+1]));e.sim.forward();e.scene.update(e.sim.mj_model.opt.timestep)
  for key,attr in [('joint_pos','joint_pos'),('joint_vel','joint_vel'),('body_pos_w','body_link_pos_w'),('body_quat_w','body_link_quat_w'),('body_lin_vel_w','body_link_lin_vel_w'),('body_ang_vel_w','body_link_ang_vel_w')]:log[key].append(getattr(robot.data,attr)[0].cpu().numpy().copy())
 np.savez_compressed(path,fps=50.,**{k:np.array(v) for k,v in log.items()});return len(tt)

assert not os.environ.get('BM_WEIGHT_MAP'),'Unified inference forbids task-based weight maps'
current_checkpoint=os.environ.get('BM_CHECKPOINT','/home/pku/frankenmotion/work/beyondmimic_demo.pt')
results=[]
with torch.inference_mode():
 for row in rows:
  stem=Path(row['path']).stem;dest=OUT/(stem+'.json')
  if dest.exists():results.append(json.loads(dest.read_text()));continue
  try:
   ref=np.load(Path(os.environ.get('BM_REFERENCE_DIR',str(ROOT/'physical_development_eval/gmr_probe')))/(stem+'_uniform.npz'))['reference_qpos'];mp=OUT/(stem+'_motion.npz');entry=1. if os.environ.get('BM_ENTRY')=='standing' else 0.;conversion_ref=ref
   if entry:
    a=np.arange(20)/20.;a=a*a*(3-2*a);lead=np.repeat(ref[:1],20,axis=0);lead[:,7:]=tr.rt.Q0+a[:,None]*(ref[0,7:]-tr.rt.Q0);conversion_ref=np.r_[lead,ref]
   if os.environ.get('BM_CHECK_KINEMATICS'):
    from mjlab_cpu_fk_20261003 import marker_parity
    err=marker_parity(e,robot,conversion_ref,model,tr);assert err<1e-5,('Robot marker geometry mismatch',err);(OUT/(stem+'_marker_parity.json')).write_text(json.dumps(dict(max_marker_error_m=err)))
   n=convert(conversion_ref,mp)
   motion.motion=MotionLoader(str(mp),motion.body_indexes,device='cuda:0');obs,_=env.reset();e.sim.forward();e.scene.update(e.sim.mj_model.opt.timestep)
   # Refresh relative target cache at frame zero without advancing reference time.
   motion.time_steps[:]=-1;motion._update_command()
   if entry:
    d=tr.mujoco.MjData(model);d.qpos[:]=conversion_ref[0];tr.rt.floor_align(model,d);robot.write_root_state_to_sim(tensor(np.r_[d.qpos[:7],np.zeros(6)][None]));robot.write_joint_state_to_sim(tensor(d.qpos[7:][order][None]),tensor(np.zeros((1,29))));e.sim.forward();e.scene.update(e.sim.mj_model.opt.timestep);motion.time_steps[:]=-1;motion._update_command()
   obs=env.get_observations();initial=np.r_[robot.data.root_link_pos_w[0].cpu().numpy(),robot.data.root_link_quat_w[0].cpu().numpy(),robot.data.joint_pos[0].cpu().numpy()[inverse]];states=[initial];errors=[];fall=None;reasons=[]
   for i in range(n-1):
    obs,rew,done,info=env.step(policy(obs))
    if bool(done.any()):
     fall=(i+1)*.02;reasons=[k for k in e.termination_manager.active_terms if bool(e.termination_manager.get_term(k).any())];break
    states.append(np.r_[robot.data.root_link_pos_w[0].cpu().numpy(),robot.data.root_link_quat_w[0].cpu().numpy(),robot.data.joint_pos[0].cpu().numpy()[inverse]])
    errors.append({k:float(v.mean()) for k,v in motion.metrics.items()})
   states=np.array(states);actual=None
   if fall is None and len(states)>1:
    ts=np.arange(len(states))*.02;ps=tr.get_positions(model,states);want=entry+np.arange(len(ref))*.05;valid=want[(want>=ts[0])&(want<=ts[-1])];p=np.stack([np.interp(valid,ts,c) for c in ps.reshape(len(ps),-1).T],1).reshape(-1,24,3);actual={'physical_only': True, 'scope': 'Exploratory mocap reference tracking; no numeric command or semantic success claim'}
   r=dict(task=row['task'],command=row.get('command'),source=row['source'],seed=row.get('seed'),actual=actual,termination_time=fall,termination_criterion=os.environ.get('BM_TERMINATION','original_tracking'),termination_reasons=reasons,steps=len(states),target_steps=n,tracking_errors={k:float(np.mean([x[k] for x in errors])) for k in errors[0]} if errors else {},initialization='standing + 1s entry' if entry else 'reference pose (RSI)',checkpoint=current_checkpoint,command_index=row.get('command_index'),input_path=row['path'])
   dest.write_text(json.dumps(r,indent=2,default=lambda x:x.item()));np.savez_compressed(OUT/(stem+'_actual.npz'),qpos=states);results.append(r);print(len(results),'requests complete' if os.environ.get('BM_QUIET_METRICS') else r,flush=True)
  except Exception as exc:
   r=dict(task=row['task'],command=row.get('command'),source=row['source'],seed=row.get('seed'),command_index=row.get('command_index'),actual=None,error=repr(exc),input_path=row['path']);dest.write_text(json.dumps(r));results.append(r);print('request_error',stem,repr(exc),flush=True)
(OUT/'results.json').write_text(json.dumps(results,indent=2,default=lambda x:x.item()));env.close()
