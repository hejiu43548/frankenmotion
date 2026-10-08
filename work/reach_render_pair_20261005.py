"""Synchronous 0.3/0.5 command comparison from saved physical states, at real speed."""
import os,json,argparse
os.environ['MUJOCO_GL']='egl'
from pathlib import Path
import numpy as np,mujoco,imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
p=argparse.ArgumentParser();p.add_argument('--near',required=True);p.add_argument('--far',required=True);p.add_argument('--output',required=True);a=p.parse_args();out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);runs=[Path(a.near),Path(a.far)];records=[]
F=lambda n:ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',n);B=lambda n:ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',n)
for r in runs:
 z=np.load(r/'actual.npz');res=json.loads((r/'result.json').read_text());meta=json.loads((r.parent/'reference_contact.json').read_text());m=mujoco.MjModel.from_binary_path(str(r/'scene.mjb'));m.vis.global_.offwidth=960;m.vis.global_.offheight=720;d=mujoco.MjData(m);cam=mujoco.MjvCamera();sc=res['scene'];cam.lookat[:]=[*sc['goal_xy'],.8];cam.lookat[:2]+=.2*np.array([np.cos(sc['direction_rad']),np.sin(sc['direction_rad'])]);cam.distance=2.55;cam.azimuth=110+np.degrees(sc['direction_rad']);cam.elevation=-24
 for g in range(m.ngeom):
  if m.geom(g).name=='terrain':m.geom_matid[g]=-1;m.geom_rgba[g]=[.16,.20,.24,1]
 records.append(dict(z=z,res=res,meta=meta,m=m,d=d,cam=cam,renderer=mujoco.Renderer(m,width=960,height=720),contacts=json.loads((r/'contacts.json').read_text()),metrics=json.loads((r/'command_metrics.json').read_text())))
assert records[0]['res']['scene']['seed']==records[1]['res']['scene']['seed'];assert records[0]['res']['scene']['walk_source']==records[1]['res']['scene']['walk_source']
start=records[0]['meta']['segments']['reach_place_lower'][0]*2.5;end=records[0]['meta']['segments']['retract_lower'][1]*2.5;indices=range(max(0,int(start)-25),min(len(records[0]['z']['qpos']),int(end)+50),2);samples=[]
with imageio.get_writer(out,fps=25,codec='libx264',quality=8) as writer:
 for frame,i in enumerate(indices):
  canvas=Image.new('RGB',(1920,848),'#101b2a');draw=ImageDraw.Draw(canvas)
  for k,r in enumerate(records):
   z=r['z'];j=min(i,len(z['qpos'])-1);m=r['m'];d=r['d'];d.qpos[:]=z['qpos'][j];mujoco.mj_forward(m,d);r['renderer'].update_scene(d,camera=r['cam']);canvas.paste(Image.fromarray(r['renderer'].render()),(k*960,80));draw=ImageDraw.Draw(canvas);cmd=r['res']['scene']['reach_command_human_m'];draw.text((k*960+30,15),f'REACH COMMAND  {cmd:.2f} m',font=B(30),fill='#63dfc2');draw.text((k*960+30,53),'Same scene / same noise / same tracker',font=F(18),fill='white');fwd=np.array([np.cos(r['res']['scene']['direction_rad']),np.sin(r['res']['scene']['direction_rad']),0]);w=m.joint('robot/right_wrist_yaw_joint').id;root=m.body('robot/pelvis').id;measured=np.dot(d.xanchor[w]-d.xpos[root],fwd)*1.2701193988323212/1.0486437524221748;touch=any(c.get('top_surface') and c['normal_force']>.2 for c in r['contacts'][j]);draw.rectangle((k*960+18,721,k*960+940,785),fill='#142231');draw.text((k*960+32,730),f'Actual wrist reach  {measured:.3f} m',font=B(23),fill='white');draw.text((k*960+590,730),'TOP CONTACT' if touch else 'FREE HAND',font=B(21),fill='#63dfc2' if touch else '#b5cadb');draw.text((k*960+32,763),'Human-equivalent units (G1 wrist measured relative to pelvis)',font=F(15),fill='#b5cadb')
  draw.line((960,0,960,800),fill='#52667e',width=3);draw.text((25,811),'Generated reach → GMR → one shared tracker | MuJoCo physics | 1x playback',font=F(18),fill='white');writer.append_data(np.asarray(canvas))
  if frame in [12,37,62,87,112,137,162,187]:samples.append(canvas.resize((960,420)))
for r in records:r['renderer'].close()
sheet=Image.new('RGB',(1920,420*((len(samples)+1)//2)))
for k,im in enumerate(samples):sheet.paste(im,((k%2)*960,(k//2)*420))
sheet.save(out.with_suffix('.png'));out.with_suffix('.json').write_text(json.dumps(dict(runs=[str(r) for r in runs],playback_speed=1.,fps=25,frames=len(indices),command_units='human-equivalent forward wrist relative to pelvis',no_robot_state_edits=True),indent=2));print(out)
