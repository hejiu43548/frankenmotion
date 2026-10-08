import os,json,argparse
os.environ['MUJOCO_GL']='egl'
from pathlib import Path
import numpy as np,mujoco
from PIL import Image,ImageDraw,ImageFont
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--segment',choices=['walk','exit'],default='walk');a=p.parse_args();run=Path(a.run);z=np.load(run/'actual.npz');r=json.loads((run/'result.json').read_text());meta=json.loads((run.parent/'reference_contact.json').read_text());lo,hi=meta['segments'][a.segment];indices=np.linspace(lo*2.5+10,min(hi*2.5-10,len(z['qpos'])-1),24).astype(int);m=mujoco.MjModel.from_binary_path(str(run/'scene.mjb'));m.vis.global_.offwidth=400;m.vis.global_.offheight=360;d=mujoco.MjData(m);cam=mujoco.MjvCamera();cam.distance=2.3;cam.azimuth=(0 if a.segment=='exit' and r['scene']['exit_task']=='sidestep' else 90)+np.degrees(r['scene']['direction_rad']);cam.elevation=-18;sheet=Image.new('RGB',(2400,1440));f=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
with mujoco.Renderer(m,width=400,height=360) as renderer:
 for k,i in enumerate(indices):
  d.qpos[:]=z['qpos'][i];mujoco.mj_forward(m,d);cam.lookat[:]=d.qpos[:3];cam.lookat[2]=.65;renderer.update_scene(d,camera=cam);im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im);draw.rectangle((0,0,400,30),fill='#101b2a');draw.text((10,6),f'{a.segment} / {r["scene"]["exit_task"]} / {i*.02:.2f}s',font=f,fill='white');sheet.paste(im,((k%6)*400,(k//6)*360))
sheet.save(run/(a.segment+'_strip.png'));print(run/(a.segment+'_strip.png'))
