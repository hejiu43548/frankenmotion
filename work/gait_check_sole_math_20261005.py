import sys,json
from pathlib import Path
from types import SimpleNamespace as NS
import torch,numpy as np,mujoco
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/gait_demo_20261005';sys.path.insert(0,str(R/'work'));from gait_rewards_20261005 import sole_data
P=D/'dev_timed/scene_000/feet_v1_1000';m=mujoco.MjModel.from_binary_path(str(P/'scene.mjb'));d=mujoco.MjData(m);z=np.load(P/'actual.npz');ref=np.load(P/'motion.npz');names=['left_ankle_roll_link','right_ankle_roll_link'];b=[m.body('robot/'+n).id for n in names];allb=[i for i in range(m.nbody) if m.body(i).name.startswith('robot/')];ri=[allb.index(i) for i in b];samples=np.linspace(20,len(z['qpos'])-1,35,dtype=int);pos=[];quat=[];native=[];floor=m.geom('terrain').id;gs=[[g for g in range(m.ngeom) if m.geom(g).name.startswith('robot/'+side+'_foot') and 'collision' in m.geom(g).name] for side in ['left','right']]
for i in samples:
 d.qpos[:]=z['qpos'][i];mujoco.mj_forward(m,d);pos.append(d.xpos[b].copy());quat.append(d.xquat[b].copy());native.append([min(mujoco.mj_geomDistance(m,d,g,floor,1.,None) for g in group) for group in gs])
t=lambda x:torch.tensor(np.asarray(x),dtype=torch.float32);term=NS(cfg=NS(body_names=names),body_pos_w=t(ref['body_pos_w'][samples][:,ri]),body_quat_w=t(ref['body_quat_w'][samples][:,ri]),robot_body_pos_w=t(pos),robot_body_quat_w=t(quat));env=NS(sim=NS(mj_model=m),command_manager=NS(get_term=lambda _:term),device='cpu',num_envs=len(samples));target,actual,arm,ids=sole_data(env);error=float(np.max(abs(actual.numpy()-native)));assert error<2e-7,error
(D/'sole_geometry_check.json').write_text(json.dumps(dict(samples=len(samples),max_clearance_error_vs_native_m=error,passed=True,formula='Min transformed capsule endpoint height minus radius; checked against native mj_geomDistance for both feet'),indent=2));print(error)
