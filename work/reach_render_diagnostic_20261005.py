import os,json,argparse
os.environ['MUJOCO_GL']='egl'
from pathlib import Path
import mujoco,numpy as np
from PIL import Image,ImageDraw,ImageFont
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();run=Path(a.run);meta=json.loads((run.parent/'reference_contact.json').read_text());z=np.load(run/'actual.npz');ref=np.load(run/'motion.npz');c=json.loads((run/'inference_contract.json').read_text());m=mujoco.MjModel.from_binary_path(str(run/'scene.mjb'));d=mujoco.MjData(m);qa=[int(m.joint('robot/'+n).qposadr[0]) for n in c['joint_names']];m.vis.global_.offwidth=600;m.vis.global_.offheight=480;cam=mujoco.MjvCamera();cam.distance=2.45;cam.azimuth=110;cam.elevation=-13;times=[0,.8,1.8,2.9,4.5,5.95];start=meta['segments']['reach_place_lower'][0];sheet=Image.new('RGB',(3600,960));font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',19)
with mujoco.Renderer(m,height=480,width=600) as renderer:
 for col,t in enumerate(times):
  phase=int((start+t*20)*2.5);phase=min(phase,len(ref['joint_pos'])-1);i=int(np.argmin(abs(z['phases']-phase)))
  for row in range(2):
   d.qpos[:]=z['qpos'][i]
   if row==0:d.qpos[:3]=ref['body_pos_w'][phase,0];d.qpos[3:7]=ref['body_quat_w'][phase,0];d.qpos[qa]=ref['joint_pos'][phase]
   mujoco.mj_forward(m,d);cam.lookat[:]=d.qpos[:3]+[.2,0,-.04];renderer.update_scene(d,camera=cam);im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im);draw.rectangle((0,0,600,42),fill='#101b2a');draw.text((10,8),('GMR reference' if row==0 else 'Physical actual')+f' / reach t={t:.2f}s',font=font,fill='white');sheet.paste(im,(col*600,row*480))
sheet.save(run/'reference_actual.png');print(run/'reference_actual.png')
