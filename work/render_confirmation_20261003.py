"""Fixed examples: prompt0/noise0/high command, archived states only."""
import os,sys,json
os.environ['MUJOCO_GL']='egl'
from pathlib import Path
BASE=Path('/home/pku/frankenmotion/outputs_amass/franken_eleven_20261003');OUT=BASE.parent/'franken_improve_20261003';sys.path.insert(0,str(BASE/'code'))
import transfer as tr
import numpy as np
import imageio.v2 as imageio
folder=OUT/'confirmation_videos';folder.mkdir(exist_ok=True);m=tr.rt.load_model();d=tr.mujoco.MjData(m);metadata=[]
for task in ['sidestep','back_walk']:
 stem=task+'_p0_s0_c4';src=OUT/'beyondmimic_confirmation'/(stem+'_actual.npz');r=json.loads((src.parent/(stem+'.json')).read_text());states=np.load(src)['qpos'];cam=tr.mujoco.MjvCamera();cam.lookat[:]=[states[:,0].mean(),states[:,1].mean(),.8];span=np.ptp(states[:,:2],axis=0).max();cam.distance=max(3.,span*1.3+1.5);cam.azimuth=135;cam.elevation=-20
 with tr.mujoco.Renderer(m,height=480,width=640) as renderer:
  with imageio.get_writer(folder/(stem+'.mp4'),fps=25,codec='libx264',quality=8) as writer:
   for i in range(0,len(states),2):
    d.qpos[:]=states[i];tr.mujoco.mj_forward(m,d);renderer.update_scene(d,camera=cam);pixels=renderer.render();writer.append_data(pixels)
    if i==2*(len(states)//4):imageio.imwrite(folder/(stem+'.png'),pixels)
 metadata.append(dict(task=task,command=r['command'],actual=r['actual'],source=str(src),selection='predeclared prompt0/noise0/highest command; no cherry-picking by score',rendering='saved actual qpos, 25fps, original physical time; SONIC mesh model with marker geometry parity to BM',termination_time=r['termination_time']))
(folder/'metadata.json').write_text(json.dumps(metadata,indent=2));print('Rendered fixed sidestep/backward confirmation examples')
