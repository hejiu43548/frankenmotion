"""Assemble the frozen V3 development references, never read confirmation inputs."""
import os,sys,json,hashlib
from pathlib import Path
os.environ['MUJOCO_GL']='egl';os.environ['BM_CPU_FK']='1'
import numpy as np
import torch
from scipy.spatial.transform import Rotation,Slerp
import mjlab.tasks
from mjlab.tasks.registry import load_env_cfg
from mjlab.envs import ManagerBasedRlEnv
R=Path('/home/pku/frankenmotion');N=R/'outputs_amass/franken_improve_20261003';U=R/'outputs_amass/franken_unified_20261004';B=R/'outputs_amass/franken_eleven_20261003'
sys.path[:0]=[str(R/'work'),str(B/'code')]
import transfer as tr
OUT=U/'development';OUT.mkdir(parents=True,exist_ok=False)
torch.set_num_threads(2);cfg=load_env_cfg('Mjlab-Tracking-Flat-Unitree-G1',play=True);cfg.scene.num_envs=1;cfg.commands['motion'].motion_file=str(R/'work/beyondmimic_demo_motion.npz');cfg.sim.nconmax=256;cfg.sim.njmax=2048
e=ManagerBasedRlEnv(cfg=cfg,device='cuda:0');robot=e.scene['robot'];model=tr.rt.load_model();names=[model.joint(i).name for i in range(1,model.njnt)];order=[names.index(n) for n in robot.joint_names]
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

read=lambda p:json.loads(p.read_text());rows=read(N/'physical_development_manifest.json');kick=read(N/'kick_old_development_manifest.json');jump=read(N/'jump_aligned_development_manifest.json');rows=[r for r in rows if r['task'] not in ['kick','jump']]+kick+jump
assert len(rows)==55;records=[];clips=[]
for row in rows:
 task=row['task'];stem=Path(row['path']).stem
 assert row['source'].endswith('_p0_s0') and 990700<=row['seed']<=990710
 if task in ['raise_hand','lean']:refpath=N/'task_retarget_v2_development'/(stem+'_w4.npz')
 elif task=='kick':refpath=N/'kick_old_retarget_development'/(stem+'_w0.npz')
 elif task=='jump':refpath=N/'jump_aligned_development_eval/gmr_probe'/(stem+'_uniform.npz')
 else:refpath=N/'physical_development_eval/gmr_probe'/(stem+'_uniform.npz')
 ref=np.load(refpath)['reference_qpos'];np.savez_compressed(OUT/(stem+'_uniform.npz'),reference_qpos=ref)
 a=np.arange(20)/20.;a=a*a*(3-2*a);lead=np.repeat(ref[:1],20,axis=0);lead[:,7:]=tr.rt.Q0+a[:,None]*(ref[0,7:]-tr.rt.Q0);path=OUT/(stem+'_motion.npz');convert(np.r_[lead,ref],path);clip=dict(np.load(path));clips.append(clip);records.append(dict(row,reference_path=str(refpath),motion_path=str(path),reference_sha256=hashlib.sha256(refpath.read_bytes()).hexdigest(),frames=len(clip['joint_pos'])))
ends=np.cumsum([x['frames'] for x in records]);np.savez_compressed(OUT/'training_motions.npz',fps=50.,**{k:np.concatenate([x[k] for x in clips]) for k in clips[0] if k!='fps'})
(OUT/'manifest.json').write_text(json.dumps(records,indent=2));(OUT/'clips.json').write_text(json.dumps(dict(ends=ends.tolist(),records=records,source='55 original development references; frozen V3 generator/retarget route; no confirmation samples'),indent=2));e.close();print('Prepared 55 independent clips',flush=True)
