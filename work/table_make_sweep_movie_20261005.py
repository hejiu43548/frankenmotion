import json,hashlib
from pathlib import Path
import imageio.v2 as imageio
import numpy as np
from PIL import Image,ImageDraw,ImageFont
D=Path('/home/pku/frankenmotion/outputs_amass/table_demo_20261005');out=D/'command_sweep';assert (out/'complete.json').exists();rows=json.loads((out/'results.json').read_text());assert len(rows)==6
readers=[imageio.get_reader(D/x['case']['name']/'scene_000/unified/presentation.mp4') for x in rows];counts=[r.count_frames() for r in readers];assert len(set(counts))==1
F=lambda s:ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',s);B=lambda s:ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf',s)
video=out/'direction_distance_demo.mp4'
with imageio.get_writer(video,fps=25,codec='libx264',quality=8) as writer:
 for frame in range(counts[0]):
  canvas=Image.new('RGB',(1920,880),'#101b2a');draw=ImageDraw.Draw(canvas);draw.text((28,12),'DISTANCE & DIRECTION COMMANDS → PHYSICAL TABLE CONTACT',font=B(28),fill='white');draw.text((28,50),'Same generation noise within each row · one frozen tracker · full-speed MuJoCo execution',font=F(21),fill='#b5cadb')
  for i,reader in enumerate(readers):
   x=(i%3)*640;y=94+(i//3)*380;canvas.paste(Image.fromarray(reader.get_data(frame)).resize((640,360)),(x,y));c=rows[i]['case'];label=f"{'DIRECTION' if i<3 else 'DISTANCE'}  |  {c['distance']:.2f} m / {np.degrees(c['angle']):+.1f}°";draw=ImageDraw.Draw(canvas);draw.rectangle((x,y,x+639,y+45),fill='#101b2a');draw.text((x+16,y+8),label,font=B(19),fill='#63dfc2');draw.text((x+530,y+12),f'{frame/25:.2f} s',font=F(17),fill='white')
  draw.text((28,855),'Systematic command demonstration; separate from the 32 random-layout evaluation. Known table pose; explicit scene-aware reach planning.',font=F(18),fill='#b5cadb');writer.append_data(np.asarray(canvas))
  if frame in [0,counts[0]//2,counts[0]-1]:canvas.save(out/f'grid_frame_{frame:03d}.png')
for r in readers:r.close()
(out/'video_metadata.json').write_text(json.dumps(dict(frames=counts[0],fps=25,playback_speed=1.,selection='All six predeclared systematic scenes, no omissions',sha256=hashlib.sha256(video.read_bytes()).hexdigest()),indent=2))
print(video)
