"""Presentation of an archived generated relative-command physical rollout."""
import os
os.environ['MUJOCO_GL']='egl'
import argparse,json,hashlib
from pathlib import Path
import numpy as np,mujoco,imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);a=p.parse_args();run=Path(a.run);out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);r=json.loads((run/'result.json').read_text());actual=np.load(run/'actual.npz');q=actual['qpos'];initial=np.load(run/'initial_motion.npz');m=mujoco.MjModel.from_binary_path(str(run/'scene.mjb'));d=mujoco.MjData(m);m.vis.global_.offwidth=1280;m.vis.global_.offheight=720
for i in range(m.ngeom):
 if m.geom(i).name=='target_mark':assert m.geom_contype[i]==0 and m.geom_conaffinity[i]==0;m.geom_rgba[i,3]=0
 if m.geom(i).name=='terrain':m.geom_matid[i]=-1;m.geom_rgba[i]=[.16,.20,.24,1]
plan=initial['body_pos_w'][:,0,:2];allxy=np.r_[plan,q[:,:2]];center=(allxy.min(0)+allxy.max(0))/2;span=float(np.ptp(allxy,axis=0).max());camera=mujoco.MjvCamera();camera.lookat[:]=[*center,.65];camera.distance=max(3.7,span+2);camera.azimuth=135;camera.elevation=-28;font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';bold='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf';F=lambda s:ImageFont.truetype(font,s);B=lambda s:ImageFont.truetype(bold,s);indices=list(range(0,len(q),2));samples={0,len(indices)-1}|{min(len(indices)-1,round((s['start50']+s['end50'])/4)) for s in r['segments']};keys=[]
def label(s):
 if s['task']=='walk':return f"WALK {s['command']:.2f} m | direction {np.degrees(s['direction_rad']):+.0f} deg"
 if s['task']=='turn':return f"TURN RIGHT {s['command']:.0f} deg"
 if s['task']=='back_walk':return f"BACKWARD WALK | {s['command']:.3f} m/s (human-equivalent)"
 if s['task']=='sidestep':return f"RIGHT SIDESTEP | {s['command']:.2f} m (human-equivalent)"
 if s['task']=='wave':return f"WAVE | amplitude {s['command']:.2f} m (human-equivalent)"
 return s['task'].upper()
with mujoco.Renderer(m,height=720,width=1280) as renderer,imageio.get_writer(str(out),fps=25,codec='libx264',quality=8) as writer:
 for ordinal,i in enumerate(indices):
  d.qpos[:]=q[i];mujoco.mj_forward(m,d);renderer.update_scene(d,camera=camera);im=Image.fromarray(renderer.render());draw=ImageDraw.Draw(im);draw.rectangle((0,0,1280,91),fill='#101b2a');draw.text((24,14),'FrankenMotion | Walk, turn and gesture',font=B(27),fill='white');draw.text((25,54),'GENERATED RELATIVE COMMANDS  /  ONE SHARED TRACKER  /  MUJOCO  /  1x SPEED',font=F(16),fill='#b5cadb');draw.text((1110,23),f'{i/50:.2f}s',font=B(23),fill='#63dfc2')
  active=next((s for s in r['segments'] if s['start50']<=i<=s['end50']),None);text=label(active) if active else 'INITIAL STANDING ENTRY' if i<r['segments'][0]['start50'] else 'FINAL SETTLE' if i>r['segments'][-1]['end50'] else 'TRANSITION';draw.rectangle((0,650,1280,720),fill='#101b2a');draw.text((24,670),text,font=B(23),fill='white');draw.text((968,675),'Actual physical rollout',font=F(17),fill='#b5cadb');draw.rectangle((0,646,int(1280*(i+1)/len(q)),649),fill='#63dfc2')
  x0,y0,w,h=24,111,292,235;draw.rounded_rectangle((x0,y0,x0+w,y0+h),radius=10,fill='#142231');draw.text((x0+12,y0+10),'INITIAL REFERENCE / ACTUAL',font=B(13),fill='#c7d9e8');lo=allxy.min(0)-.2;extent=allxy.max(0)+.2-lo;scale=min((w-35)/max(extent[0],.1),(h-55)/max(extent[1],.1))
  def xy(p):return (x0+16+(p[0]-lo[0])*scale,y0+h-18-(p[1]-lo[1])*scale)
  draw.line([xy(p) for p in plan],fill='#78889a',width=2)
  if i:draw.line([xy(p) for p in q[:i+1,:2]],fill='#63dfc2',width=3)
  px,py=xy(q[i,:2]);draw.ellipse((px-4,py-4,px+4,py+4),fill='#63dfc2');draw.text((x0+12,y0+h-19),'Relative stage commands; no perception',font=F(12),fill='#b5cadb')
  if not r['physical_complete'] and i==indices[-1]:draw.text((425,120),'ROLLOUT TERMINATED',font=B(26),fill='#ffb68f')
  writer.append_data(np.asarray(im))
  if ordinal in samples:keys.append(im.resize((640,360)))
canvas=Image.new('RGB',(1280,360*int(np.ceil(len(keys)/2))),'white')
for k,im in enumerate(keys):canvas.paste(im,((k%2)*640,(k//2)*360))
canvas.save(out.with_suffix('.png'));out.with_suffix('.json').write_text(json.dumps(dict(source_run=str(run),checkpoint_sha256=json.loads((run.parent/'protocol.json').read_text())['checkpoint_sha256'],raw_sha256=hashlib.sha256((run/'actual.npz').read_bytes()).hexdigest(),commands=[label(s) for s in r['segments']],frames=len(indices),fps=25,time_scale=1.,scope=__doc__,visual_edits='Floor colour, fixed camera, labels and 2D trajectory overlay only. No scene collision/state changes. Initial reference is gray; actual is teal. Reference anchoring is a controller-side operation already logged in source run.'),indent=2));print(out)
