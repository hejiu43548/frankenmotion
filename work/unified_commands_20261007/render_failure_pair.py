import os
os.environ['MUJOCO_GL']='egl'
from pathlib import Path
import json,numpy as np,mujoco,imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/unified_commands_20261007';out=D/'visuals/failure_pairs';out.mkdir(exist_ok=True);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
for case in ['walk_p0_s0_c4','jump_p0_s0_c4']:
 runs=[D/'general_evaluation'/name/case for name in ['baseline_final','unified_final']];zz=[np.load(run/'actual.npz') for run in runs];rr=[json.loads((run/'result.json').read_text()) for run in runs];ref=np.load(runs[0]/'motion.npz');m=mujoco.MjModel.from_binary_path(str(runs[0]/'scene.mjb'));d=mujoco.MjData(m);m.vis.global_.offwidth=640;m.vis.global_.offheight=480;camera=mujoco.MjvCamera();camera.type=mujoco.mjtCamera.mjCAMERA_FREE;camera.distance=2.6;camera.azimuth=125;camera.elevation=-18;samples=[];n=max(len(z['qpos']) for z in zz);indices=list(range(0,n,2));indices+=([n-1] if indices[-1]!=n-1 else []);keep=set(np.linspace(0,len(indices)-1,8,dtype=int))
 for gid in range(m.ngeom):
  if m.geom(gid).name=='target_mark':
   assert m.geom_contype[gid]==0 and m.geom_conaffinity[gid]==0;m.geom_rgba[gid,3]=0
 with mujoco.Renderer(m,height=480,width=640) as renderer,imageio.get_writer(str(out/(case+'.mp4')),fps=25,codec='libx264',quality=8) as writer:
  for ordinal,i in enumerate(indices):
   im=Image.new('RGB',(1280,560),'#f4f7fa');draw=ImageDraw.Draw(im);camera.lookat[:]=[*ref['body_pos_w'][min(i,len(ref['body_pos_w'])-1),0,:2],.7]
   for side in range(2):
    idx=min(i,len(zz[side]['qpos'])-1);d.qpos[:]=zz[side]['qpos'][idx];mujoco.mj_forward(m,d);renderer.update_scene(d,camera=camera);im.paste(Image.fromarray(renderer.render()),(640*side,80));label=['Original separate controls','ONE shared control'][side];draw.text((640*side+12,8),label+' | actual physics',font=font,fill='#152c42');done=not rr[side]['physical_complete'] and i>=len(zz[side]['qpos'])-1;status='TERMINATED - LAST STATE HELD' if done else 'running';draw.text((640*side+12,38),f'{case} | t={i/50:.2f}s | {status}',font=font,fill='#ba1e26' if done else '#152c42')
   writer.append_data(np.asarray(im))
   if ordinal in keep:samples.append(im.resize((640,280)))
 sheet=Image.new('RGB',(1280,280*((len(samples)+1)//2)),'white')
 for i,im in enumerate(samples):sheet.paste(im,(i%2*640,i//2*280))
 sheet.save(out/(case+'.png'))
(out/'protocol.json').write_text(json.dumps(dict(selection='Explicit failure audit: first prompt/noise high-speed walk and high jump, not success-selected.',after_termination='Last recorded actual state held, explicit red label; no simulated continuation or reference replacement.',runs=['baseline_final','unified_final'],fps=25),indent=2))
