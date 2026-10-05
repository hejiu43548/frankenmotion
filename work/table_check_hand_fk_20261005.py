import os,json,sys
os.environ['MUJOCO_GL']='egl'
from pathlib import Path
import mujoco,numpy as np
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/table_demo_20261005/development_v2'
for i in range(6):
 scene=D/f'scene_{i:03d}';run=scene/'interaction_v1_0500';m=mujoco.MjModel.from_binary_path(str(run/'scene.mjb'));d=mujoco.MjData(m);z=np.load(run/'motion.npz');ref=np.load(scene/'reference_lift.npz')['reference_qpos'][-1];names=[m.joint(j).name.removeprefix('robot/') for j in range(1,m.njnt)]
 sys.path.insert(0,str(R/'outputs_amass/franken_eleven_20261003/code'));import transfer as tr
 source=tr.rt.load_model();order=[source.joint(n).qposadr[0] for n in names];d.qpos[:7]=ref[:7];d.qpos[7:]=ref[order];mujoco.mj_forward(m,d);target=json.loads((scene/'scene.json').read_text())['hand_target'];hand=m.geom('robot/right_hand_collision').id
 print(i,'native_refhand',d.geom_xpos[hand], 'target',target,'error',np.linalg.norm(d.geom_xpos[hand]-target))
