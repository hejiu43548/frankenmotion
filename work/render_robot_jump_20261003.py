"""Render saved physical G1 rollout; never resimulate or alter states."""
import os,sys
os.environ['MUJOCO_GL']='egl'
from pathlib import Path
BASE=Path('/home/pku/frankenmotion/outputs_amass/franken_eleven_20261003');OUT=BASE.parent/'franken_improve_20261003';sys.path.insert(0,str(BASE/'code'))
import transfer as tr
import numpy as np
import imageio.v2 as imageio
m=tr.rt.load_model();d=tr.mujoco.MjData(m);folder=OUT/'visuals_jump_aligned';folder.mkdir(exist_ok=True)
z=np.load(OUT/'jump_aligned_development_eval/gmr_probe/jump_p0_s0_c4_uniform.npz');states=z['qpos'];ts=z['time_s'];cam=tr.mujoco.MjvCamera();cam.lookat[:]=[states[0,0],states[0,1],.9];cam.distance=3.;cam.azimuth=135;cam.elevation=-12
with tr.mujoco.Renderer(m,height=480,width=640) as renderer:
 with imageio.get_writer(folder/'g1_jump_055_actual.mp4',fps=25,codec='libx264',quality=8) as writer:
  for i in range(0,len(states),2):
   d.qpos[:]=states[i];tr.mujoco.mj_forward(m,d);renderer.update_scene(d,camera=cam);pixels=renderer.render();writer.append_data(pixels)
   if abs(ts[i]-2.5)<.025:imageio.imwrite(folder/'g1_jump_055_actual_peak.png',pixels)
print('Rendered',len(states),'physical samples',flush=True)
