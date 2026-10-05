"""Compose a funding demo from full-speed audited physical rollouts."""
import json,subprocess,hashlib
from pathlib import Path
import numpy as np,imageio.v2 as imageio
from PIL import Image,ImageDraw,ImageFont
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/table_demo_20261005';assert (D/'final_pipeline_complete.json').exists();summary=json.loads((D/'final_test/unified_summary.json').read_text());results=summary['results'];assert len(results)==32;out=D/'videos';out.mkdir(exist_ok=False);hero=next(r for r in results if r['success']);indexes=sorted(set([0,1,2,3,hero['scene']['index']]))
for i in indexes:
 run=Path(results[i]['scene']['source'])/'unified';assert json.loads((run/'audit.json').read_text())['success']==results[i]['success'];subprocess.run([str(R/'work/mjlab_stable_env/bin/python'),str(R/'work/table_render_presentation_20261005.py'),'--run',str(run),'--output',str(out/f'scene_{i:03d}.mp4')],cwd=R,check=True)
font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';bold='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf';F=lambda s:ImageFont.truetype(font,s);B=lambda s:ImageFont.truetype(bold,s)
def center(draw,y,text,font,color='white'):
 box=draw.textbbox((0,0),text,font=font);draw.text(((1280-box[2])/2,y),text,font=font,fill=color)
def card():return Image.new('RGB',(1280,720),'#101b2a')
intro=card();d=ImageDraw.Draw(intro);center(d,90,'GOAL → MOTION → CONTACT',B(44));center(d,158,'FrankenMotion × Unitree G1',F(30),'#63dfc2');center(d,235,'A randomly placed table becomes a distance-and-direction goal.',F(22),'#d3e1eb')
labels=['KNOWN\nTABLE POSE','GOAL-CONDITIONED\nGENERATION','GMR +\nSCENE-AWARE IK','ONE SHARED\nTRACKER','MUJOCO\nPHYSICS']
for i,label in enumerate(labels):
 x=40+i*248;d.rounded_rectangle((x,350,x+218,445),radius=12,fill='#1e3448',outline='#38647a',width=2);lines=label.split('\n')
 for j,line in enumerate(lines):
  b=d.textbbox((0,0),line,font=B(16));d.text((x+(218-b[2])/2,372+j*25),line,font=B(16),fill='white')
 if i<4:d.text((x+224,380),'›',font=B(27),fill='#63dfc2')
center(d,525,'Physical contacts · full-speed execution · same policy in every scene',F(21),'#b5cadb');center(d,620,'Simulation prototype  |  Known table pose',F(18),'#85a3b9')
ending=card();d=ImageDraw.Draw(ending);center(d,70,'HELD-OUT RANDOM LAYOUTS',B(30),'#b5cadb');center(d,130,f'{summary["successes"]} / 32 successful interactions',B(43),'#63dfc2');center(d,245,f'Mean walking-goal error   {summary["mean_root_error"]*100:.1f} cm',F(27));center(d,295,f'Mean hand-target error   {summary["mean_hand_error"]*100:.1f} cm',F(27));center(d,375,'Success: stable completion + target accuracy + sustained tabletop contact',F(19),'#b5cadb');center(d,418,'Walk goals 0.85–1.75 m / ±28.6° · tabletop 0.80 m · fixed walk/reach prompt templates',F(17),'#85a3b9');center(d,525,'Next milestone: broader scenes and real-robot validation',B(24));center(d,620,'Learned goal conditioning + explicit scene planning + a shared full-body controller',F(18),'#b5cadb')
hero_path=out/f'scene_{hero["scene"]["index"]:03d}.mp4';readers=[imageio.get_reader(out/f'scene_{i:03d}.mp4') for i in range(4)];counts=[r.count_frames() for r in readers];grid_count=max(counts);movie=out/'funding_demo.mp4'
with imageio.get_writer(movie,fps=25,codec='libx264',quality=8) as writer:
 for _ in range(100):writer.append_data(np.array(intro))
 with imageio.get_reader(hero_path) as reader:
  for frame in reader:writer.append_data(frame)
 with imageio.get_writer(out/'first_four_random_layouts.mp4',fps=25,codec='libx264',quality=8) as gridwriter:
  for frame in range(grid_count):
   grid=Image.new('RGB',(1280,720),'#101b2a')
   for i,reader in enumerate(readers):
    tile=Image.fromarray(reader.get_data(min(frame,counts[i]-1))).resize((640,360))
    if not results[i]['success']:
     draw=ImageDraw.Draw(tile);draw.rectangle((15,310,410,343),fill='#803234');draw.text((24,316),'FAILED TRIAL — retained in evaluation',font=F(15),fill='white')
    grid.paste(tile,((i%2)*640,(i//2)*360))
   gridwriter.append_data(np.array(grid));writer.append_data(np.array(grid))
   if frame==grid_count-1:grid.save(out/'first_four_final_frame.png')
 for _ in range(150):writer.append_data(np.array(ending))
for reader in readers:reader.close()
intro.save(out/'intro.png');ending.save(out/'results_card.png');metadata=dict(hero_index=hero['scene']['index'],hero_selection='First successful final trial by ordered index; all 32 trials retained in benchmark. Montage always shows first four, including any failures.',physics_playback_speed=1.,fps=25,intro_frames=100,outro_frames=150,full_hero_duration=hero['duration_s'],final_successes=summary['successes'],final_planned=32,source_runs=[str(Path(results[i]['scene']['source'])/'unified') for i in indexes],files={p.name:dict(bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in out.glob('*.mp4')});(out/'metadata.json').write_text(json.dumps(metadata,indent=2));print(movie,flush=True)
