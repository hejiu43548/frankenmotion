"""Deterministic all-eleven evidence video: first fresh source, middle command, failures retained."""
import os
os.environ['MUJOCO_GL']='egl'
import json,hashlib
from pathlib import Path
import numpy as np,mujoco,imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
D=Path('/home/pku/frankenmotion/outputs_amass/universal_tracker_20261006');out=D/'visuals/eleven_grid';out.mkdir(exist_ok=False)
rows=json.loads((D/'general_evaluation/fresh_candidate/audited_results.json').read_text());selected=[r for r in rows if r['source']==r['task']+'_p0_s0' and r['command_index']==2];assert len(selected)==11
names=['Raise hand','Forward reach','Strike speed','Wave amplitude','Right turn','Right sidestep','Backward walk','Right kick','Jump height','Forward lean','Forward walk'];units=['m','m','m/s','m','rad','m','m/s','m','m','rad','m/s'];tols=[.02,.02,.1,.015,.08,.06,.05,.05,.04,.06,.05];font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';bold='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf';F=lambda s:ImageFont.truetype(font,s);B=lambda s:ImageFont.truetype(bold,s);records=[]
for row,name,unit,tol in zip(selected,names,units,tols):
 run=Path(row['run']);q=np.load(run/'actual.npz')['qpos'];m=mujoco.MjModel.from_binary_path(str(run/'scene.mjb'));d=mujoco.MjData(m)
 for i in range(m.ngeom):
  if m.geom(i).name=='target_mark':assert m.geom_contype[i]==0 and m.geom_conaffinity[i]==0;m.geom_rgba[i,3]=0
  if m.geom(i).name=='terrain':m.geom_matid[i]=-1;m.geom_rgba[i]=[.16,.20,.24,1]
 cam=mujoco.MjvCamera();cam.distance=2.7;cam.azimuth=135;cam.elevation=-20
 passed=row['actual'] is not None and row['actual']['event_pass'] and abs(row['actual']['quantity']-row['command'])<=tol
 path=out/(row['task']+'.mp4');count=0
 with mujoco.Renderer(m,height=272,width=400) as renderer,imageio.get_writer(str(path),fps=25,codec='libx264',quality=8,macro_block_size=1) as writer:
  for i in range(0,len(q),2):
   d.qpos[:]=q[i];mujoco.mj_forward(m,d);cam.lookat[:]=[q[i,0],q[i,1],.70];renderer.update_scene(d,camera=cam);im=Image.new('RGB',(400,350),'#101b2a');im.paste(Image.fromarray(renderer.render()),(0,48));dr=ImageDraw.Draw(im);dr.text((10,5),name,font=B(18),fill='white');dr.text((10,27),f"Command {row['command']:.3g} {unit} | t={i/50:.2f}s",font=F(14),fill='#c2d3e1');qval=row['actual']['quantity'] if row['actual'] else float('nan');dr.rectangle((0,320,400,350),fill='#101b2a');dr.text((10,328),f"Q={qval:.3f} {unit} | {'PASS' if passed else 'OUTSIDE CRITERIA'}",font=B(13),fill='#63dfc2' if passed else '#ffb68f');writer.append_data(np.asarray(im));count+=1
 records.append(dict(task=row['task'],source=row['source'],command=row['command'],actual=row['actual'],joint_pass=passed,video=str(path),frames=count,raw_sha256=hashlib.sha256((run/'actual.npz').read_bytes()).hexdigest(),run=str(run)));print(row['task'],passed,flush=True)
readers=[imageio.get_reader(r['video']) for r in records];last=[None]*11;maximum=max(r['frames'] for r in records);samples=[]
with imageio.get_writer(str(out/'eleven_actions.mp4'),fps=25,codec='libx264',quality=8,macro_block_size=1) as writer:
 for i in range(maximum):
  canvas=Image.new('RGB',(1600,1110),'#101b2a');dr=ImageDraw.Draw(canvas);dr.text((18,13),'ONE FROZEN TRACKER | 11 ACTIONS | FIRST SOURCE / MIDDLE COMMAND | 1x SPEED',font=B(24),fill='white')
  for j,(reader,r) in enumerate(zip(readers,records)):
   if i<r['frames']:last[j]=Image.fromarray(reader.get_data(i))
   im=last[j].copy()
   if i>=r['frames']:ImageDraw.Draw(im).text((240,27),'FINISHED / HELD',font=B(13),fill='#ffdd94')
   canvas.paste(im,((j%4)*400,60+(j//4)*350))
  x,y=1215,805
  for k,line in enumerate(['Actual MuJoCo rollouts','Same shared MLP, no routing','All outcomes retained','Human-equivalent units','Angles remain radians','Q: full-clip audited quantity','No per-clip time warping']):dr.text((x,y+k*29),line,font=F(17),fill='#c2d3e1')
  writer.append_data(np.asarray(canvas))
  if i in [0,maximum//3,2*maximum//3,maximum-1]:samples.append(canvas.resize((800,555)))
for reader in readers:reader.close()
montage=Image.new('RGB',(1600,1110))
for j,im in enumerate(samples):montage.paste(im,((j%2)*800,(j//2)*555))
montage.save(out/'inspection_montage.png');(out/'protocol.json').write_text(json.dumps(dict(scope=__doc__,selection='Fixed first prompt p0 / first noise s0 / middle command c2 for every class, no success-based selection.',checkpoint_sha256=json.loads((D/'frozen_unified/protocol.json').read_text())['checkpoint_sha256'],fps=25,time_scale=1,records=records),indent=2))
