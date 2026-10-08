"""A one-minute presentation reel of two complete clips containing three frozen-policy physical trials."""
from pathlib import Path
import json,hashlib
import numpy as np,imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
D=Path('/home/pku/frankenmotion/outputs_amass/universal_tracker_20261006');out=D/'visuals/presentation_reel';out.mkdir(exist_ok=False);font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';bold='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf';F=lambda n:ImageFont.truetype(font,n);B=lambda n:ImageFont.truetype(bold,n);size=(1600,900);fps=25
clips=[D/'visuals/frozen_candidates/fresh_table_paired.mp4',D/'visuals/frozen_service_trimmed/service_0.mp4'];records=[];preview=[]
def card(title,subtitle,footer):
 im=Image.new('RGB',size,'#101b2a');dr=ImageDraw.Draw(im);dr.rectangle((110,260,1490,266),fill='#63dfc2');dr.text((110,330),title,font=B(59),fill='white');dr.text((110,425),subtitle,font=F(31),fill='#c1d5e5');dr.text((110,775),footer,font=F(23),fill='#7dcdbf');return im
first=card('FrankenMotion to G1','ONE SHARED TRACKER  /  GENERATED MOTION COMMANDS','Two separate MuJoCo scenes  |  Complete trajectories  |  1x speed')
transition=card('SCENE 2','Walk, point, bow, turn and leave','Same frozen tracker  |  Text-conditioned point and bow  |  Numeric walk and turn')
last=card('A shared policy across both scenes','Offline generated references  /  Physical simulation','G1 29-DoF body with fixed hands  |  Full metrics and limitations in the report')
count=0
with imageio.get_writer(str(out/'one_minute_demo.mp4'),fps=fps,codec='libx264',quality=8,macro_block_size=1) as writer:
 for _ in range(50):writer.append_data(np.asarray(first));count+=1
 preview.append(first.resize((800,450)))
 for j,path in enumerate(clips):
  if j:
   for _ in range(38):writer.append_data(np.asarray(transition));count+=1
  reader=imageio.get_reader(str(path));meta=reader.get_meta_data();assert meta['fps']==fps;n=0;begin=count
  for frame in reader:
   im=Image.fromarray(frame);im.thumbnail(size,Image.Resampling.LANCZOS);canvas=Image.new('RGB',size,'#101b2a');canvas.paste(im,((size[0]-im.width)//2,(size[1]-im.height)//2))
   if j==0:
    dr=ImageDraw.Draw(canvas);dr.text((35,40),'SCENE 1  |  SAME LAYOUT, SAME TRACKER, DIFFERENT REACH COMMAND',font=B(28),fill='white');dr.text((35,815),'Command: 0.30 / 0.50 m   |   Actual held reach: 0.325 / 0.523 m (human-equivalent)',font=F(27),fill='#c1d5e5')
   writer.append_data(np.asarray(canvas));count+=1;n+=1
   if n==250:preview.append(canvas.resize((800,450)))
  reader.close();records.append(dict(source=str(path),source_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),frames=n,reel_start_frame=begin,time_scale=1.,full_clip_included=True))
 for _ in range(50):writer.append_data(np.asarray(last));count+=1
 preview.append(last.resize((800,450)))
montage=Image.new('RGB',(1600,900))
for j,im in enumerate(preview):montage.paste(im,((j%2)*800,(j//2)*450))
montage.save(out/'one_minute_demo.png');(out/'one_minute_demo.json').write_text(json.dumps(dict(scope=__doc__,checkpoint_sha256=json.loads((D/'frozen_unified/protocol.json').read_text())['checkpoint_sha256'],fps=fps,frames=count,duration_s=count/fps,clips=records,edits='Title cards, explicit scene cut, aspect-preserving resize and letterboxing. No trajectory changes, no time warping, no within-clip omissions.',selection='Presentation selection from previously audited and visually checked candidates. Not a new benchmark.'),indent=2));print('Reel duration',count/fps,flush=True)
