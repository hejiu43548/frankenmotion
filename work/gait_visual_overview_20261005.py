import os,json
os.environ['MUJOCO_GL']='egl'
from pathlib import Path
import mujoco,numpy as np
from PIL import Image,ImageDraw,ImageFont
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/gait_demo_20261005';out=D/'visual_review';out.mkdir(exist_ok=True);rows=json.loads((D/'final_random32/manifest.json').read_text());font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',16)
for page in range(8):
 sheet=Image.new('RGB',(2000,1400))
 for rr,row in enumerate(rows[page*4:page*4+4]):
  s=Path(row['source']);run=s/'selected';z=np.load(run/'actual.npz');res=json.loads((run/'result.json').read_text());meta=json.loads((s/'reference_contact.json').read_text());lo,hi=np.array(meta['segments']['walk'])*2.5;phase=[lo+(hi-lo)*.25,lo+(hi-lo)*.50,lo+(hi-lo)*.75,hi+45,z['phases'][-1]];inds=[int(np.argmin(abs(z['phases']-p))) for p in phase];m=mujoco.MjModel.from_binary_path(str(run/'scene.mjb'));d=mujoco.MjData(m);m.vis.global_.offwidth=500;m.vis.global_.offheight=350;cam=mujoco.MjvCamera();cam.distance=2.5;cam.azimuth=110+np.degrees(row['direction_rad']);cam.elevation=-12
  with mujoco.Renderer(m,height=350,width=500) as rend:
   for cc,i in enumerate(inds):
    d.qpos[:]=z['qpos'][i];mujoco.mj_forward(m,d);cam.lookat[:]=d.qpos[:3]+[.1,0,-.06];rend.update_scene(d,camera=cam);im=Image.fromarray(rend.render());dr=ImageDraw.Draw(im);dr.rectangle((0,0,500,43),fill='#101b2a');dr.text((8,4),f'Scene {row["index"]:02d} | {i*.02:.2f}s | success={res["success"]}',font=font,fill='white');dr.text((8,23),f'{row["distance_robot_m"]:.2f}m / {np.degrees(row["direction_rad"]):+.1f}deg',font=font,fill='white');
    if cc<4:sheet.paste(im,((cc)*500,rr*350))
    else:im.save(out/f'contact_{row["index"]:03d}.png')
 sheet.save(out/f'walk_stop_{page:02d}.png')
contact=Image.new('RGB',(2000,2800))
for i in range(32):contact.paste(Image.open(out/f'contact_{i:03d}.png'),((i%4)*500,(i//4)*350))
contact.save(out/'all_contacts.png');print(out)
