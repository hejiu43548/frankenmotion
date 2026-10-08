"""Dense actual-pose strip for visual gait inspection; camera follows actual pelvis."""
import os
os.environ['MUJOCO_GL']='egl'
import json,argparse,hashlib
from pathlib import Path
import numpy as np,mujoco
from PIL import Image,ImageDraw,ImageFont
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);p.add_argument('--start',type=float,default=1.);p.add_argument('--end',type=float,default=4.);p.add_argument('--count',type=int,default=20);a=p.parse_args();run=Path(a.run);out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True)
z=np.load(run/'actual.npz');states=z['qpos'];m=mujoco.MjModel.from_binary_path(str(run/'scene.mjb'));d=mujoco.MjData(m);cam=mujoco.MjvCamera();cam.type=mujoco.mjtCamera.mjCAMERA_FREE;cam.distance=2.3;cam.azimuth=90;cam.elevation=-15;indices=np.unique(np.linspace(round(a.start*50),min(len(states)-1,round(a.end*50)),a.count).astype(int));font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16);w,h=240,350;columns=10;canvas=Image.new('RGB',(columns*w,int(np.ceil(len(indices)/columns))*h),'#f7f9fb');renderer=mujoco.Renderer(m,height=h-30,width=w)
for name in ['target_mark']:
 gid=mujoco.mj_name2id(m,mujoco.mjtObj.mjOBJ_GEOM,name)
 if gid>=0:
  assert m.geom_contype[gid]==0 and m.geom_conaffinity[gid]==0;m.geom_rgba[gid,3]=0
for j,i in enumerate(indices):
 d.qpos[:]=states[i];mujoco.mj_forward(m,d);cam.lookat[:]=[states[i,0],states[i,1],.68];renderer.update_scene(d,camera=cam);im=Image.fromarray(renderer.render());x=(j%columns)*w;y=(j//columns)*h;canvas.paste(im,(x,y+30));ImageDraw.Draw(canvas).text((x+8,y+5),f'Actual | {i/50:.2f}s',font=font,fill='#203e54')
renderer.close();canvas.save(out);out.with_suffix('.json').write_text(json.dumps(dict(run=str(run),raw_sha256=hashlib.sha256((run/'actual.npz').read_bytes()).hexdigest(),indices=indices.tolist(),times_s=(indices/50).tolist(),scope=__doc__,caution='Discrete visual samples, not a claim of watching the full video or proof of absence of all jitter.'),indent=2));print(out)
