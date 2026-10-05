import os,sys,json,hashlib
from pathlib import Path
os.environ['MUJOCO_GL']='egl'
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/table_demo_20261005';U=R/'outputs_amass/franken_unified_20261004';B=R/'outputs_amass/franken_eleven_20261003';sys.path[:0]=[str(R/'work'),str(B/'code')]
import torch,numpy as np,mjlab.tasks
from scipy.spatial.transform import Rotation,Slerp
from mjlab.tasks.registry import load_env_cfg
from mjlab.envs import ManagerBasedRlEnv
from mjlab_cpu_fk_20261003 import convert_cpu
import transfer as tr
out=D/'training_corpus_v1';out.mkdir(exist_ok=False);torch.set_num_threads(2);cfg=load_env_cfg('Mjlab-Tracking-Flat-Unitree-G1',play=True);cfg.scene.num_envs=1;cfg.commands['motion'].motion_file=str(R/'work/beyondmimic_demo_motion.npz');cfg.sim.nconmax=256;cfg.sim.njmax=2048;e=ManagerBasedRlEnv(cfg=cfg,device='cuda:0');robot=e.scene['robot'];m=tr.rt.load_model();names=[m.joint(i).name for i in range(1,m.njnt)];order=[names.index(n) for n in robot.joint_names]
records=json.loads((U/'training_corpus_augmented/manifest.json').read_text());clips=[dict(np.load(x['motion_path'])) for x in records];accepted=[];rejected=[]
for row in json.loads((D/'training_v2/manifest.json').read_text()):
 folder=Path(row['source']);meta=json.loads((folder/'reference_lift.json').read_text())
 if meta['hand_ik_final_error_m']>.03:rejected.append(dict(scene=row['index'],error=meta['hand_ik_final_error_m']));continue
 states=np.load(folder/'reference_lift.npz')['reference_qpos'];t=np.arange(len(states))*.05;tt=np.arange(int(np.ceil(t[-1]/.02))+1)*.02;ts=np.minimum(tt,t[-1]);pos=np.stack([np.interp(ts,t,c) for c in states[:,:3].T],1);rots=Slerp(t,Rotation.from_quat(states[:,[4,5,6,3]]))(ts);quat=rots.as_quat()[:,[3,0,1,2]];q=np.stack([np.interp(ts,t,c) for c in states[:,7:].T],1)[:,order];dq=np.gradient(q,.02,axis=0);vel=np.gradient(pos,.02,axis=0);av=(rots[1:]*rots[:-1].inv()).as_rotvec()/.02;av=np.r_[av,av[-1:]];path=out/f'table_{row["index"]:03d}_motion.npz';convert_cpu(e,robot,pos,quat,q,dq,vel,av,rots,path);clip=dict(np.load(path));clips.append(clip);records.append(dict(task='table_approach',seed=row['seed'],motion_path=str(path),frames=len(q),scene_metadata=row,reference_sha256=hashlib.sha256((folder/'reference_lift.npz').read_bytes()).hexdigest()));accepted.append(row['index']);print('converted',len(accepted),flush=True)
ends=np.cumsum([len(x['joint_pos']) for x in clips]);np.savez_compressed(out/'training_motions.npz',fps=50.,**{k:np.concatenate([x[k] for x in clips]) for k in clips[0] if k!='fps'});(out/'manifest.json').write_text(json.dumps(records,indent=2));(out/'clips.json').write_text(json.dumps(dict(ends=ends.tolist(),records=records),indent=2));(out/'audit.json').write_text(json.dumps(dict(prior_clips=735,new_accepted=accepted,new_rejected=rejected,selection='training-only geometrical IK residual <=3cm; no development/final trajectories',frames=int(ends[-1]),sampling='half environments new table clips; half original tasks'),indent=2));e.close();print('corpus complete',len(records),flush=True)
