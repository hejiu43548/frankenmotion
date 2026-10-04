"""All-task development marker parity between native mjlab and SONIC robot models."""
import os,sys,json
os.environ['MUJOCO_GL']='egl'
from pathlib import Path
import numpy as np
import torch
import mjlab.tasks
from mjlab.tasks.registry import load_env_cfg
from mjlab.envs import ManagerBasedRlEnv
ROOT=Path('/home/pku/frankenmotion/outputs_amass/franken_improve_20261003');BASE=ROOT.parent/'franken_eleven_20261003';sys.path.insert(0,str(BASE/'code'));import transfer as tr
from mjlab_cpu_fk_20261003 import marker_parity
cfg=load_env_cfg('Mjlab-Tracking-Flat-Unitree-G1',play=True);cfg.scene.num_envs=1;cfg.commands['motion'].motion_file='/home/pku/frankenmotion/work/beyondmimic_demo_motion.npz';torch.set_num_threads(2);e=ManagerBasedRlEnv(cfg=cfg,device='cuda:0');robot=e.scene['robot'];m=tr.rt.load_model();records=[]
paths=list((ROOT/'physical_development_eval/gmr_probe').glob('*_uniform.npz'))+list((ROOT/'jump_aligned_development_eval/gmr_probe').glob('*_uniform.npz'))
for path in paths:
 states=np.load(path)['reference_qpos'];error=marker_parity(e,robot,states,m,tr);assert error<1e-6,(str(path),error);records.append(dict(path=str(path),frames=len(states),max_marker_error_m=error))
r=dict(cases=len(records),frames=sum(x['frames'] for x in records),max_marker_error_m=max(x['max_marker_error_m'] for x in records),records=records);(ROOT/'marker_geometry_audit.json').write_text(json.dumps(r,indent=2));print({k:v for k,v in r.items() if k!='records'});e.close()
