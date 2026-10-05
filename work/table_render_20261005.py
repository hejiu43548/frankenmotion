import os,json,argparse
os.environ['MUJOCO_GL']='egl'
from pathlib import Path
import mujoco,numpy as np,imageio.v2 as imageio,zipfile
from PIL import Image,ImageDraw,ImageFont
def load_model(out):
 if (out/'scene.mjb').exists():return mujoco.MjModel.from_binary_path(str(out/'scene.mjb'))
 with zipfile.ZipFile(out/'scene.zip') as z:
  name=next(n for n in z.namelist() if n.endswith('.xml'));xml=z.read(name).decode().replace('<default/>','').replace('material="silver"','material="robot/silver"');assets={n.removeprefix('assets/'):z.read(n) for n in z.namelist() if n.startswith('assets/')}
 return mujoco.MjModel.from_xml_string(xml,assets)
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();out=Path(a.run);res=json.loads((out/'result.json').read_text());z=np.load(out/'actual.npz');q=z['qpos'];model=load_model(out);data=mujoco.MjData(model);meta=res['scene'];cam=mujoco.MjvCamera();cam.lookat[:]=[meta['table_center'][0]/2,meta['table_center'][1]/2,.75];cam.distance=3.8;cam.azimuth=135+np.degrees(meta['direction_rad']);cam.elevation=-18;frames=[]
model.vis.global_.offwidth=1280;model.vis.global_.offheight=720
with mujoco.Renderer(model,height=720,width=1280) as renderer:
 with imageio.get_writer(out/'actual.mp4',fps=25,codec='libx264',quality=8) as writer:
  for i in range(0,len(q),2):
   data.qpos[:]=q[i];mujoco.mj_forward(model,data);renderer.update_scene(data,camera=cam);im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im);draw.rectangle([0,0,1280,66],fill='#0b1423');draw.text((20,12),f'PHYSICS ROLLOUT | One shared tracker | t = {(i+1)*.02:.2f} s',fill='white');draw.text((20,37),f'Goal: {meta["distance_robot_m"]:.2f} m / {np.degrees(meta["direction_rad"]):+.1f} deg | Scene {meta["index"]} | {a.run.split("/")[-1]}',fill='#85dcc3');writer.append_data(np.array(im))
   if i in [0,100,250,400,550,len(q)//2*2-2]:frames.append(im.resize((640,360)))
canvas=Image.new('RGB',(640*3,360*2),'white')
for k,im in enumerate(frames[:6]):canvas.paste(im,((k%3)*640,(k//3)*360))
canvas.save(out/'contact_sheet.png');print(out/'actual.mp4')
