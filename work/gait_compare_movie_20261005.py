from pathlib import Path
import imageio.v2 as imageio,numpy as np,json
from PIL import Image,ImageDraw,ImageFont
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/gait_demo_20261005';out=D/'videos';out.mkdir(exist_ok=True);sources=[R/'outputs_amass/table_demo_20261005/development_v3/scene_000/selected_fresh/walk_inspection.mp4',D/'dev_timed/scene_000/distill_v2_3000_gpu/walk_inspection.mp4'];rd=[imageio.get_reader(p) for p in sources];ns=[r.count_frames() for r in rd];font=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',24);small=ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf',18)
with imageio.get_writer(out/'gait_before_after.mp4',fps=25,codec='libx264',quality=8) as wr:
 for i in range(max(ns)):
  im=Image.new('RGB',(1920,800),'#101b2a');dr=ImageDraw.Draw(im)
  for k in range(2):
   tile=Image.fromarray(rd[k].get_data(min(i,ns[k]-1)));im.paste(tile,(k*960,60));dr.text((k*960+20,12),['PREVIOUS: slow reference + previous tracker','NEW: retimed reference + one distilled tracker'][k],font=font,fill='white')
   if i>=ns[k]:dr.rectangle((k*960+100,370,k*960+850,420),fill='#101b2a');dr.text((k*960+125,381),'Walking segment completed; last frame held',font=font,fill='#63dfc2')
  dr.text((20,779),'Same generated path and table layout. Both play at 1x real time; no trajectory edits in rendering.',font=small,fill='white');wr.append_data(np.asarray(im))
for r in rd:r.close()
(out/'gait_before_after.json').write_text(json.dumps(dict(sources=[str(s) for s in sources],fps=25,time_scale=1,comparison='Same development scene0 and generated spatial path; new pipeline explicitly retimes reference. Both GPU physical trajectories. Shorter clip freezes only after its walking phase is over.'),indent=2))
