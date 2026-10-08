import os,json,argparse
os.environ['MUJOCO_GL']='egl'
from pathlib import Path
import numpy as np,mujoco
from PIL import Image,ImageDraw,ImageFont
p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--output',required=True);a=p.parse_args();folder=Path(a.folder);out=Path(a.output);out.mkdir(exist_ok=True);rows=json.loads((folder/'manifest.json').read_text());font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15)
for group in range(0,len(rows),4):
 canvas=Image.new('RGB',(1600,1200),'#101b2a')
 for k,row in enumerate(rows[group:group+4]):
  s=Path(row['source']);run=s/'selected';z=np.load(run/'actual.npz');meta=json.loads((s/'reference_contact.json').read_text());res=json.loads((run/'result.json').read_text());seg=meta['segments'];wanted=[np.mean(seg['walk']),np.mean(seg['contact_hold']),seg['retract_lower'][1]-2,np.mean(seg['exit'])];labels=['APPROACH','CONTACT HOLD','HAND DOWN','DEPARTURE'];idx=[int(np.argmin(abs(z['phases'][:len(z['qpos'])]-t*2.5))) for t in wanted];m=mujoco.MjModel.from_binary_path(str(run/'scene.mjb'));m.vis.global_.offwidth=400;m.vis.global_.offheight=300;d=mujoco.MjData(m);cam=mujoco.MjvCamera();cam.distance=2.7;cam.azimuth=90+np.degrees(row['direction_rad']);cam.elevation=-20
  with mujoco.Renderer(m,width=400,height=300) as rr:
   for j,i in enumerate(idx):
    d.qpos[:]=z['qpos'][i];mujoco.mj_forward(m,d);cam.lookat[:]=d.qpos[:3];cam.lookat[2]=.65;rr.update_scene(d,camera=cam);im=Image.fromarray(rr.render());draw=ImageDraw.Draw(im);draw.rectangle((0,0,400,45),fill='#101b2a');draw.text((8,3),f"Case{row['index']:02d} / reach{row['command']:.2f} / {labels[j]}",font=font,fill='white');draw.text((8,24),f"{row['exit_task']} / full-stage success={res['success']}",font=font,fill='#63dfc2');canvas.paste(im,(j*400,k*300))
 canvas.save(out/f'all_cases_{group:02d}_{min(group+3,len(rows)-1):02d}.png')
