"""Physical MuJoCo-Warp table interaction with one frozen shared tracker."""
import os,sys,json,argparse,hashlib,dataclasses
from pathlib import Path
os.environ['MUJOCO_GL']='egl';os.environ['WANDB_MODE']='disabled'
R=Path('/home/pku/frankenmotion');sys.path[:0]=[str(R/'work'),str(R/'outputs_amass/franken_eleven_20261003/code')]
import numpy as np,torch,mujoco,mjlab.tasks
from scipy.spatial.transform import Rotation,Slerp
from scipy.optimize import least_squares
from mjlab.tasks.registry import load_env_cfg,load_rl_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import MjlabOnPolicyRunner,RslRlVecEnvWrapper
from mjlab.managers import TerminationTermCfg
from mjlab.tasks.tracking.mdp.commands import MotionLoader
from unified_preview_20261004 import configure_preview,checkpoint_offsets
from mjlab_cpu_fk_20261003 import convert_cpu
import transfer as tr
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--name',default='sim_v1');p.add_argument('--reference',default='reference');p.add_argument('--checkpoint',default=str(R/'outputs_amass/franken_unified_20261004/frozen_unified/policy.pt'));p.add_argument('--pace',action='store_true');p.add_argument('--hand-feedback',action='store_true');p.add_argument('--fresh-initial-observation',action='store_true');p.add_argument('--render',action='store_true');a=p.parse_args();scene=Path(a.scene).resolve();meta=json.loads((scene/'scene.json').read_text());refmeta=json.loads((scene/(a.reference+'.json')).read_text());out=scene/a.name;out.mkdir(exist_ok=False);torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.1);torch.manual_seed(61005)

def add_table(spec):
 yaw=meta['table_yaw'];center=np.array(meta['table_center']);top=meta['table_top'];half=meta['table_half_size'];body=spec.worldbody.add_body(name='demo_table',pos=[center[0],center[1],0],quat=Rotation.from_euler('z',yaw).as_quat()[[3,0,1,2]])
 body.add_geom(name='table_top',type=mujoco.mjtGeom.mjGEOM_BOX,size=half,pos=[0,0,top-half[2]],rgba=[.50,.28,.12,1],contype=1,conaffinity=1,friction=[.8,.02,.002])
 for i,x in enumerate([-.32,.32]):
  for j,y in enumerate([-.40,.40]):body.add_geom(name=f'table_leg_{i}{j}',type=mujoco.mjtGeom.mjGEOM_BOX,size=[.025,.025,(top-.07)/2],pos=[x,y,(top-.07)/2],rgba=[.15,.19,.24,1],contype=1,conaffinity=1)
 # Visible target disk is non-colliding and is never used to determine success.
 spec.worldbody.add_geom(name='target_mark',type=mujoco.mjtGeom.mjGEOM_CYLINDER,size=[.055,.0005,0],pos=[*meta['hand_target'][:2],top+.0008],rgba=[.1,.8,.55,1],contype=0,conaffinity=0)

def fall(env):
 data=env.scene['robot'].data;q=data.root_link_quat_w;w,x,y,z=q.unbind(-1);roll=torch.atan2(2*(w*x+y*z),1-2*(x*x+y*y));pitch=torch.asin((2*(w*y-z*x)).clamp(-1,1));return (data.root_link_pos_w[:,2]<.35)|((roll**2+pitch**2).sqrt()>1.05)|~torch.isfinite(data.joint_pos).all(-1)
name='Mjlab-Tracking-Flat-Unitree-G1';cfg=load_env_cfg(name,play=True);cfg.scene.num_envs=1;cfg.scene.spec_fn=add_table;cfg.scene.extent=5.;cfg.events={};cfg.sim.nconmax=256;cfg.sim.njmax=2048;cfg.commands['motion'].motion_file=str(R/'work/beyondmimic_demo_motion.npz');cfg.commands['motion'].joint_position_range=(0.,0.);cfg.terminations={'physical_fall':TerminationTermCfg(func=fall)};offsets=checkpoint_offsets(a.checkpoint);configure_preview(cfg,offsets)
e=ManagerBasedRlEnv(cfg=cfg,device='cuda:0');robot=e.scene['robot'];motion=e.command_manager.get_term('motion');agent=load_rl_cfg(name);env=RslRlVecEnvWrapper(e,clip_actions=agent.clip_actions);runner=MjlabOnPolicyRunner(env,dataclasses.asdict(agent),device='cuda:0');runner.load(a.checkpoint,load_cfg={'actor':True},strict=True,map_location='cuda:0');policy=runner.get_inference_policy(device='cuda:0')
model=tr.rt.load_model();names=[model.joint(i).name for i in range(1,model.njnt)];order=[names.index(n) for n in robot.joint_names];inverse=np.argsort(order);ref=np.load(scene/(a.reference+'.npz'))['reference_qpos'];t=np.arange(len(ref))*.05;tt=np.arange(int(np.ceil(t[-1]/.02))+1)*.02;ts=np.minimum(tt,t[-1]);pos=np.stack([np.interp(ts,t,c) for c in ref[:,:3].T],1);rots=Slerp(t,Rotation.from_quat(ref[:,[4,5,6,3]]))(ts);quat=rots.as_quat()[:,[3,0,1,2]];q=np.stack([np.interp(ts,t,c) for c in ref[:,7:].T],1)[:,order];dq=np.gradient(q,.02,axis=0);vel=np.gradient(pos,.02,axis=0);av=(rots[1:]*rots[:-1].inv()).as_rotvec()/.02;av=np.r_[av,av[-1:]];mp=out/'motion.npz';convert_cpu(e,robot,pos,quat,q,dq,vel,av,rots,mp);motion.motion=MotionLoader(str(mp),motion.body_indexes,device='cuda:0')
tensor=lambda x:torch.as_tensor(x,dtype=torch.float32,device='cuda:0')
obs,_=env.reset();robot.write_root_state_to_sim(tensor(np.r_[pos[0],quat[0],np.zeros(6)][None]));robot.write_joint_state_to_sim(tensor(q[:1]),tensor(np.zeros((1,29))));e.sim.forward();e.scene.update(e.sim.mj_model.opt.timestep);motion.time_steps[:]=-1;motion._update_command()
if a.fresh_initial_observation:e.observation_manager.reset()
obs=env.get_observations()
# All state writes end here. Each subsequent state comes from env.step(policy(obs)).
native=e.sim.mj_model
term=e.action_manager.get_term('joint_pos')
def serial(value):return value[0].detach().cpu().tolist() if torch.is_tensor(value) and value.ndim>1 else value.detach().cpu().tolist() if torch.is_tensor(value) else value
contract=dict(joint_names=list(robot.joint_names),default_joint_pos=serial(robot.data.default_joint_pos),action_target_names=list(term.target_names),action_scale=serial(term.scale),action_offset=serial(term.offset),clip_actions=agent.clip_actions,preview_offsets=list(offsets),anchor_body_name='robot/torso_link',reference_anchor_index=list(robot.body_names).index('torso_link'),linear_velocity_sensor='robot/imu_lin_vel',angular_velocity_sensor='robot/imu_ang_vel',physics_timestep=float(native.opt.timestep),control_timestep=.02,initial_qpos=e.sim.data.qpos[0].cpu().tolist(),initial_qvel=e.sim.data.qvel[0].cpu().tolist(),observation_order=['reference_joint_pos','reference_joint_vel','anchor_position_local','anchor_orientation_first_two_columns','imu_linear_velocity','imu_angular_velocity','joint_position_minus_default','joint_velocity','previous_normalized_action','preview_offsets_each_q_dq_anchor_pos_anchor_ori'],encoder_bias=serial(robot.data.encoder_bias))
(out/'inference_contract.json').write_text(json.dumps(contract,indent=2))
data=mujoco.MjData(native);tableid=native.geom('table_top').id;handids=[i for i in range(native.ngeom) if 'right_hand_collision' in native.geom(i).name];assert len(handids)==1;handid=handids[0];states=[];qvels=[];ctrls=[];actions=[];observations=[];phases=[];contacts=[];palms=[];rootlog=[];termination=None
# Cache the unmodified world-space hand path for scene-relative feedback.
handpath=[]
ix=robot.indexing;jq=ix.joint_q_adr.cpu().numpy();fq=ix.free_joint_q_adr.cpu().numpy()
for k in range(len(tt)):
 data.qpos[fq]=np.r_[pos[k],quat[k]];data.qpos[jq]=q[k];mujoco.mj_forward(native,data);handpath.append(data.geom_xpos[handid].copy())
handpath=np.asarray(handpath);original_joint=motion.motion.joint_pos.clone();armidx=[i for i,n in enumerate(robot.joint_names) if n.startswith('right_') and any(x in n for x in ['shoulder','elbow','wrist'])];armadr=jq[armidx];jointids=ix.joint_ids.cpu().numpy()[armidx];limits=native.jnt_range[jointids];correction=np.zeros(7);feedbacklog=[]
maxsteps=len(tt)-1+(500 if a.pace else 0)
with torch.inference_mode():
 for step in range(maxsteps):
  phase=int(motion.time_steps[0]);phases.append(phase)
  if a.hand_feedback and phase>=int(refmeta['segments']['reach_hold'][0]*2.5) and step%5==0:
   state=e.sim.data.qpos[0].cpu().numpy().copy();data.qpos[:]=state;mujoco.mj_forward(native,data);actual_hand=data.geom_xpos[handid].copy();nominal=original_joint[phase].cpu().numpy();want=handpath[phase]+np.clip(.3*(handpath[phase]-actual_hand),-.05,.05);previous=nominal[armidx]+correction
   def residual(x):
    data.qpos[:]=state;data.qpos[armadr]=x;mujoco.mj_forward(native,data)
    return np.r_[15*(data.geom_xpos[handid]-want),.04*(x-nominal[armidx]),.1*(x-previous)]
   fit=least_squares(residual,np.clip(previous,limits[:,0]+1e-4,limits[:,1]-1e-4),bounds=(limits[:,0]+1e-4,limits[:,1]-1e-4),max_nfev=25)
   new=fit.x-nominal[armidx];correction+=np.clip(new-correction,-.08,.08);end=min(phase+26,len(tt));motion.motion.joint_pos[phase:end,armidx]=original_joint[phase:end,armidx]+tensor(correction)[None];obs=env.get_observations();feedbacklog.append(dict(time_s=step*.02,phase=phase,correction=correction.tolist(),target=want.tolist()))
  observations.append(obs['actor'][0].cpu().numpy().copy());act=policy(obs);actions.append(act[0].cpu().numpy());obs,rew,done,info=env.step(act)
  if bool(done.any()):termination=(step+1)*.02;break
  full=e.sim.data.qpos[0].cpu().numpy().copy();velocity=e.sim.data.qvel[0].cpu().numpy().copy();states.append(full);qvels.append(velocity);data.qpos[:]=full;data.qvel[:]=velocity;control=e.sim.data.ctrl[0].cpu().numpy().copy();ctrls.append(control);data.ctrl[:]=control;mujoco.mj_forward(native,data);palms.append(data.geom_xpos[handid].copy());rootlog.append(robot.data.root_link_pos_w[0].cpu().numpy().copy());found=[]
  for j in range(data.ncon):
   c=data.contact[j]
   if tableid in c.geom and handid in c.geom:
    force=np.zeros(6);mujoco.mj_contactForce(native,data,j,force);found.append(dict(distance=float(c.dist),normal_force=float(force[0]),position=c.pos.tolist(),top_surface=bool(c.pos[2]>meta['table_top']-.012 and abs(c.frame[2])>.8 and data.geom_xpos[handid,2]>meta['table_top'])))
  contacts.append(found)
  if int(motion.time_steps[0])>=len(tt)-2:break
  if a.pace:
   now=int(motion.time_steps[0]);distance=float(np.linalg.norm(rootlog[-1][:2]-pos[now,:2]))
   # Hold the reference clock when its root outruns the robot. No pose teleport.
   if distance>.16 and now<refmeta['segments']['reach_hold'][0]*2.5:
    motion.time_steps[:]-=2;motion._update_command();obs=env.get_observations()
states=np.asarray(states);palms=np.asarray(palms);rootlog=np.asarray(rootlog);np.savez_compressed(out/'actual.npz',qpos=states,qvel=np.asarray(qvels),ctrl=np.asarray(ctrls),observations=np.asarray(observations),actions=np.asarray(actions),phases=np.asarray(phases),palm=palms,root=rootlog,fps=50.);(out/'contacts.json').write_text(json.dumps(contacts));(out/'feedback.json').write_text(json.dumps(feedbacklog));e.scene._spec.to_zip(str(out/'scene.zip'));mujoco.mj_saveModel(native,str(out/'scene.mjb'),None)
active=np.array([any(c['normal_force']>.2 and c['top_surface'] for c in cs) for cs in contacts]);longest=run=0
for x in active:run=run+1 if x else 0;longest=max(longest,run)
result=dict(scene=meta,checkpoint=a.checkpoint,checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),single_tracker=True,physical_complete=termination is None and int(motion.time_steps[0])>=len(tt)-2,termination_time=termination,duration_s=len(states)*.02,reference_phase_final=int(motion.time_steps[0]),reference_frames=len(tt),root_final=rootlog[-1].tolist() if len(rootlog) else None,goal_error_m=float(np.linalg.norm(rootlog[-1,:2]-meta['goal_xy'])) if len(rootlog) else None,palm_final=palms[-1].tolist() if len(palms) else None,palm_target_error_m=float(np.linalg.norm(palms[-1]-meta['hand_target'])) if len(palms) else None,hand_contact_s=float(active.sum()*.02),longest_hand_contact_s=longest*.02,contact_method='Native MuJoCo mj_forward and contact solver on saved physical qpos/qvel; saved actual actuator ctrl; named hand/table pair, normal force >0.2N, upward top face and palm above table',domain_randomization=False,state_writes='Initial state only; all subsequent states produced by shared policy and physics',pace=a.pace,hand_feedback=a.hand_feedback,fresh_initial_observation=a.fresh_initial_observation)
result['success']=bool(result['physical_complete'] and result['goal_error_m']<.2 and result['palm_target_error_m']<.1 and result['longest_hand_contact_s']>=1.)
(out/'result.json').write_text(json.dumps(result,indent=2));print(json.dumps(result,indent=2));e.close()
