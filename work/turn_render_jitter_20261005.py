import os,json
os.environ['MUJOCO_GL']='egl'
from pathlib import Path
import numpy as np,mujoco,imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
R=Path('/home/pku/frankenmotion/outputs_amass');D=R/'turn_demo_20261005';scene=R/'reach_demo_20261005/final_paired16/scene_011';runs=[scene/'selected',scene/'turn_full_actor_diagnostic'];z=[np.load(r/'actual.npz') for r in runs];meta=json.loads((scene/'reference_contact.json').read_text());sc=json.loads((scene/'scene.json').read_text());lo,hi=meta['segments']['walk'];m=mujoco.MjModel.from_binary_path(str(runs[0]/'scene.mjb'));m.vis.global_.offwidth=800;m.vis.global_.offheight=640;d=mujoco.MjData(m);cam=mujoco.MjvCamera();cam.distance=2.6;cam.azimuth=90+np.degrees(sc['direction_rad']);cam.elevation=-16;cam.lookat[:]=[*((z[0]['root'][int(lo*2.5),:2]+z[0]['root'][int(hi*2.5)-1,:2])/2),.73];font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',24);samples=[];out=D/'videos/walking_before_after.mp4'
with mujoco.Renderer(m,width=800,height=640) as rr,imageio.get_writer(out,fps=50,codec='libx264',quality=8) as writer:
 for i in range(int(lo*2.5),int(hi*2.5)):
  im=Image.new('RGB',(1600,720),'#101b2a');draw=ImageDraw.Draw(im)
  for k,label in enumerate(['PREVIOUS REACH TRACKER','REPAIRED SINGLE TRACKER']):
   d.qpos[:]=z[k]['qpos'][i];mujoco.mj_forward(m,d);rr.update_scene(d,camera=cam);im.paste(Image.fromarray(rr.render()),(k*800,60));draw.text((k*800+25,19),label,font=font,fill='white')
  writer.append_data(np.asarray(im))
  if (i-int(lo*2.5))%30==0:samples.append(im.resize((800,360)))
canvas=Image.new('RGB',(1600,1080),'white')
for k,im in enumerate(samples[:6]):canvas.paste(im,((k%2)*800,(k//2)*360))
canvas.save(out.with_suffix('.png'));out.with_suffix('.json').write_text(json.dumps(dict(source_runs=list(map(str,runs)),fps=50,time_scale=1.,reference_identical=True,camera='Same fixed world camera; no follow-camera masking of root motion.',render='Unmodified saved physical qpos, no temporal smoothing.'),indent=2));print(out)
