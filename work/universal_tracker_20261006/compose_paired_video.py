"""Side-by-side real-time physical replays; no trajectory editing or time warping."""
import argparse,json,hashlib
from pathlib import Path
import numpy as np,imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
p=argparse.ArgumentParser();p.add_argument('--left',required=True);p.add_argument('--right',required=True);p.add_argument('--output',required=True);p.add_argument('--title',default='Generated reach command | 0.30 m vs 0.50 m');a=p.parse_args();paths=[Path(a.left),Path(a.right)];readers=[imageio.get_reader(str(p)) for p in paths];metadata=[r.get_meta_data() for r in readers];assert all(abs(m['fps']-25)<1e-5 for m in metadata);counts=[r.count_frames() for r in readers];out=Path(a.output);out.parent.mkdir(parents=True,exist_ok=True);font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',23);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',15);last=[None,None]
with imageio.get_writer(str(out),fps=25,codec='libx264',quality=8,macro_block_size=1) as writer:
 for i in range(max(counts)):
  canvas=Image.new('RGB',(1600,520),'#101b2a');draw=ImageDraw.Draw(canvas);draw.text((20,9),a.title,font=font,fill='white');draw.text((20,40),'One shared policy | separate physical rollouts | synchronized start | 1x speed',font=small,fill='#b5cadb')
  for side,reader in enumerate(readers):
   if i<counts[side]:last[side]=Image.fromarray(reader.get_data(i)).resize((800,450),Image.Resampling.LANCZOS)
   canvas.paste(last[side],(side*800,70))
   if i>=counts[side]:ImageDraw.Draw(canvas).text((side*800+18,95),'ROLLOUT FINISHED',font=font,fill='#ffda85')
  writer.append_data(np.asarray(canvas))
for r in readers:r.close()
out.with_suffix('.json').write_text(json.dumps(dict(inputs=[str(p) for p in paths],input_sha256=[hashlib.sha256(p.read_bytes()).hexdigest() for p in paths],input_frames=counts,fps=25,time_scale=1.,output_frames=max(counts),title=a.title,scope=__doc__,padding='Shorter video holds last frame with a visible ROLLOUT FINISHED label.'),indent=2));print(out)
