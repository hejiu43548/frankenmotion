"""Presentation rendering of archived physics only, with measured contact HUD."""
import os,json,argparse
os.environ['MUJOCO_GL']='egl'
from pathlib import Path
import mujoco,numpy as np,imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output');p.add_argument('--label',default='FrankenMotion | Commanded reach');a=p.parse_args();run=Path(a.run);out=Path(a.output) if a.output else run/'presentation.mp4';out.parent.mkdir(parents=True,exist_ok=True);res=json.loads((run/'result.json').read_text());z=np.load(run/'actual.npz');q=z['qpos'];root=z['root'];phases=z['phases'];contacts=json.loads((run/'contacts.json').read_text());scene=res['scene'];model=mujoco.MjModel.from_binary_path(str(run/'scene.mjb'));data=mujoco.MjData(model);ref=np.load(run/'motion.npz');planned=ref['body_pos_w'][:,0,:2]
font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';bold='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf';F=lambda size:ImageFont.truetype(font,size);B=lambda size:ImageFont.truetype(bold,size)
main=mujoco.MjvCamera();main.lookat[:]=[float(np.mean([root[:,0].min(),root[:,0].max()]))+.20,float(np.mean([root[:,1].min(),root[:,1].max()])),.68];main.distance=3.6;main.azimuth=135+np.degrees(scene['direction_rad']);main.elevation=-18
close=mujoco.MjvCamera();close.lookat[:]=np.array(scene['hand_target'])+[-.05,0,.03];close.distance=1.15;close.azimuth=145+np.degrees(scene['direction_rad']);close.elevation=-24
model.vis.global_.offwidth=1280;model.vis.global_.offheight=720
# Visual-only floor styling: physical geometry and all recorded states unchanged.
for i in range(model.ngeom):
 if model.geom(i).name=='terrain':model.geom_matid[i]=-1;model.geom_rgba[i]=[.16,.20,.24,1]
reference_meta=json.loads((run.parent/'reference_contact.json').read_text());walk_end=reference_meta['segments']['walk'][1]*2.5;reach_start=reference_meta['segments']['reach_hold'][0]*2.5
canvas_samples=[];keep=set(np.clip(np.array([reference_meta['segments']['walk'][0]+20,reference_meta['segments']['walk'][1]-5,reference_meta['segments']['contact_hold'][0]+10,reference_meta['segments']['retract_lower'][0]+20,reference_meta['segments']['retract_lower'][1]-2,reference_meta['segments']['exit'][1]+15])*2.5,0,len(q)-2).astype(int)//2*2);total_contact=0.
with mujoco.Renderer(model,height=720,width=1280) as renderer, mujoco.Renderer(model,height=240,width=320) as detail:
 with imageio.get_writer(out,fps=25,codec='libx264',quality=8) as writer:
  for i in range(0,len(q),2):
   data.qpos[:]=q[i];mujoco.mj_forward(model,data);renderer.update_scene(data,camera=main);im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im)
   touch=any(c.get('top_surface',False) and c['normal_force']>.2 for c in contacts[i]);total_contact=sum(any(c.get('top_surface',False) and c['normal_force']>.2 for c in row) for row in contacts[:i+1])*.02
   draw.rectangle((0,0,1280,92),fill='#101b2a');draw.text((28,15),a.label,font=B(28),fill='white');draw.text((28,53),'GENERATED COMMANDS  /  ONE SHARED TRACKER  /  MUJOCO PHYSICS  /  1x SPEED',font=F(16),fill='#b5cadb')
   draw.text((835,20),f"REACH {scene['reach_command_human_m']:.2f} m",font=B(27),fill='#63dfc2');draw.text((838,56),f'Human-equivalent wrist reach  |  {(i+1)*.02:.2f}s',font=F(17),fill='white')
   # Plan and measured root trajectory share the same fixed world coordinates.
   x0,y0,w,h=24,112,258,192;draw.rounded_rectangle((x0,y0,x0+w,y0+h),radius=10,fill='#142231');draw.text((x0+12,y0+9),'TOP VIEW  /  target & travel',font=B(13),fill='#c7d9e8')
   extent=max(2.6,scene['table_center'][0]+.6);sx=sy=min((w-30)/extent,(h-48)/1.5)
   def point(xy):return (x0+18+xy[0]*sx,y0+h/2+18-xy[1]*sy)
   draw.line([point(x) for x in planned],fill='#718095',width=2)
   if i>0:draw.line([point(x) for x in root[:i+1,:2]],fill='#63dfc2',width=3)
   gx,gy=point(scene['goal_xy']);draw.ellipse((gx-5,gy-5,gx+5,gy+5),outline='white',width=2);rx,ry=point(root[i,:2]);draw.ellipse((rx-4,ry-4,rx+4,ry+4),fill='#63dfc2')
   target_error=float(np.linalg.norm(root[i,:2]-scene['goal_xy']));draw.text((x0+12,y0+h-24),f"WALK {scene['distance_robot_m']:.2f}m / {np.degrees(scene['direction_rad']):+.0f}deg",font=F(14),fill='white')
   phase=phases[i] if i<len(phases) else phases[-1];stage='APPROACH' if phase<walk_end else 'SETTLE' if phase<reach_start else 'RAISE / REACH'; elapsed=(phase/2.5-reference_meta['segments']['reach_place_lower'][0])/20; stage=('PLACE HAND' if 1.8<=elapsed<2.4 else 'HOLD ON TABLE' if 2.4<=elapsed<3.6 else 'LIFT / RETRACT' if 3.6<=elapsed<5.2 else 'LOWER ARM' if 5.2<=elapsed<6 else stage); exlo,exhi=reference_meta['segments']['exit']; stage='PREPARE TO LEAVE' if elapsed>=6 and phase<exlo*2.5 else stage; stage=('BACK AWAY' if scene['exit_task']=='back_walk' else 'SIDESTEP AWAY') if exlo*2.5<=phase<exhi*2.5 else 'FINISH' if phase>=exhi*2.5 else stage
   if phase>=reach_start:
    detail.update_scene(data,camera=close);crop=Image.fromarray(detail.render());im.paste(crop,(932,382));draw=ImageDraw.Draw(im);draw.rectangle((930,380,1253,623),outline='#63dfc2',width=2);draw.rectangle((932,353,1251,380),fill='#142231');draw.text((943,359),'HAND / TABLE CONTACT',font=B(13),fill='white')
   draw.rectangle((0,654,1280,720),fill='#101b2a');draw.text((28,669),stage,font=B(22),fill='white');color='#63dfc2' if touch else '#acbfd0';status='TOP CONTACT' if touch else 'NO TOP CONTACT';draw.text((315,671),status,font=B(18),fill=color);draw.text((540,671),f'Contact time  {total_contact:.2f} s',font=F(18),fill='white');draw.text((965,672),'Actual physical rollout',font=F(16),fill='#b5cadb');draw.rectangle((0,650,int(1280*(i+1)/len(q)),653),fill='#63dfc2')
   writer.append_data(np.asarray(im))
   if i in keep:canvas_samples.append(im.resize((640,360)))
canvas=Image.new('RGB',(1920,720),'white')
for k,im in enumerate(canvas_samples[:6]):canvas.paste(im,((k%3)*640,(k//3)*360))
canvas.save(out.with_suffix('.png'));out.with_suffix('.json').write_text(json.dumps(dict(source_run=str(run),source_result=res,video_fps=25,physics_record_fps=50,time_scale=1.,every_second_physical_frame=True,visual_changes='floor colour, labels and top-view/close-up overlays only; same main camera for every scene; no robot state/trajectory changes',contact_overlay='sampled recorded top-contact flags with >0.2N recomputed force',selection_label=a.label),indent=2));print(out)
