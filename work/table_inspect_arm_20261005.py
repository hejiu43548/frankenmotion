from pathlib import Path
import numpy as np,mujoco
D=Path('/home/pku/frankenmotion/outputs_amass/table_demo_20261005/development_v2/scene_000');run=D/'interaction_v1_0500';m=mujoco.MjModel.from_binary_path(str(run/'scene.mjb'));ref=np.load(run/'motion.npz');actual=np.load(run/'actual.npz')['qpos'][-1];d=mujoco.MjData(m)
# Motion robot order is model joint order for this config; derive by saved source order explicitly.
import sys
sys.path.insert(0,'/home/pku/frankenmotion/outputs_amass/franken_eleven_20261003/code');import transfer as tr
sm=tr.rt.load_model();raw=np.load(D/'reference_lift.npz')['reference_qpos'][-1]
for j in range(1,m.njnt):
 name=m.joint(j).name.removeprefix('robot/');idx=m.joint(j).qposadr[0];want=raw[sm.joint(name).qposadr[0]]
 if 'right_' in name and any(x in name for x in ['wrist','shoulder','elbow']):print(name,round(want,3),round(actual[idx],3),round(actual[idx]-want,3))
