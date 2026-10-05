import os,json,argparse
os.environ['MUJOCO_GL']='egl'
from pathlib import Path
import numpy as np,mujoco,imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=Path(a.run);z=np.load(run/'actual.npz');meta=json.loads((run.parent/'reference_contact.json').read_text());lo,hi=meta['segments']['walk'];frames=np.flatnonzero((z['phases']>=lo*2.5)&(z['phases']<hi*2.5))[::2];m=mujoco.MjModel.from_binary_path(str(run/'scene.mjb'));d=mujoco.MjData(m);m.vis.global_.offwidth=960;m.vis.global_.offheight=720;cam=mujoco.MjvCamera();cam.distance=2.3;cam.azimuth=100;cam.elevation=-12;footcam=mujoco.MjvCamera();footcam.distance=1.05;footcam.azimuth=110;footcam.elevation=-10;f=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18);keep=set(np.linspace(0,len(frames)-1,12,dtype=int));stills=[]
with mujoco.Renderer(m,height=720,width=960) as renderer,mujoco.Renderer(m,height=250,width=400) as detail,imageio.get_writer(run/'walk_inspection.mp4',fps=25,codec='libx264',quality=8) as writer:
 for j,i in enumerate(frames):
  d.qpos[:]=z['qpos'][i];mujoco.mj_forward(m,d);cam.lookat[:]=d.qpos[:3]+[.15,0,-.08];renderer.update_scene(d,camera=cam);im=Image.fromarray(renderer.render());footcam.lookat[:]=[d.qpos[0],d.qpos[1],.23];detail.update_scene(d,camera=footcam);im.paste(Image.fromarray(detail.render()),(560,450));draw=ImageDraw.Draw(im);draw.rectangle((0,0,960,58),fill='#101b2a');draw.text((15,7),run.name+' / actual physical walking',font=f,fill='white');draw.text((15,32),f'{i*.02:.2f}s / 1x playback / feet close-up at lower right',font=f,fill='white');writer.append_data(np.asarray(im))
  if j in keep:stills.append(im.resize((480,360)))
sheet=Image.new('RGB',(1920,1080))
for j,im in enumerate(stills):sheet.paste(im,((j%4)*480,(j//4)*360))
sheet.save(run/'walk_inspection.png');print(run/'walk_inspection.png')
