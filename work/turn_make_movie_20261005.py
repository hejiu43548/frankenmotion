"""Presentation edit of complete physical rollouts and a synchronized reach comparison."""
import json,argparse,subprocess
from pathlib import Path
from PIL import Image,ImageDraw,ImageFont
import imageio_ffmpeg
p=argparse.ArgumentParser();p.add_argument('--videos',required=True);p.add_argument('--back',required=True);p.add_argument('--side',required=True);p.add_argument('--pair',required=True);a=p.parse_args();v=Path(a.videos);v.mkdir(parents=True,exist_ok=True);ff=imageio_ffmpeg.get_ffmpeg_exe();font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';bold='/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf';segments=[];details=[]
def run(args):subprocess.run([ff,'-hide_banner','-loglevel','error','-y']+args,check=True)
def card(name,title,lines,seconds):
 im=Image.new('RGB',(1280,720),'#101b2a');d=ImageDraw.Draw(im);d.rectangle((72,130,80,565),fill='#63dfc2');d.text((110,148),'FRANKENMOTION  /  G1',font=ImageFont.truetype(bold,23),fill='#63dfc2');d.text((110,216),title,font=ImageFont.truetype(bold,40),fill='white')
 for i,line in enumerate(lines):d.text((110,310+i*50),line,font=ImageFont.truetype(font,23),fill='#c1d3e3')
 d.text((110,615),'Nominal MuJoCo simulation | Known table pose | One fixed tracker',font=ImageFont.truetype(font,19),fill='#8da8be');png=v/(name+'.png');im.save(png);dest=v/(name+'.mp4');run(['-loop','1','-framerate','25','-i',str(png),'-t',str(seconds),'-an','-c:v','libx264','-pix_fmt','yuv420p','-crf','18',str(dest)]);segments.append(dest);details.append(dict(type='title',file=str(dest),duration=seconds))
def clip(source,name):
 dest=v/(name+'.mp4');run(['-i',str(source),'-vf','scale=1280:720:force_original_aspect_ratio=decrease,pad=1280:720:(ow-iw)/2:(oh-ih)/2:color=0x101b2a,setsar=1,fps=25','-an','-c:v','libx264','-pix_fmt','yuv420p','-crf','18',str(dest)]);segments.append(dest);details.append(dict(type='physical_rollout',source=str(source),file=str(dest),time_scale=1.,whole_source_clip=True))
card('title','Reach. Release. Turn. Walk away.', ['Generated distance, direction, reach and turn commands.','FrankenMotion generation → GMR → shared tracker'],3.)
card('compare_title','Same scene. Different reach commands.', ['0.30 m and 0.50 m enter the motion generator.','Reach units: human-equivalent wrist distance from pelvis.'],2.5)
clip(Path(a.pair),'comparison_segment')
if a.back:
 card('back_title','Full sequence: turn right 90 degrees', ['Approach, place hand, retract, turn, and walk away at 1x.'],2.)
 clip(Path(a.back),'back_segment')
card('side_title','Full sequence: turn right 135 degrees', ['The same tracker executes every stage.'],2.)
clip(Path(a.side),'side_segment')
card('end_title','Next: real-world validation.', ['Perception, robustness and hardware testing remain ahead.','This video demonstrates the simulation pipeline.'],3.)
listing=v/'concat.txt';listing.write_text(''.join("file '"+str(s.resolve()).replace("'","'\\''")+"'\n" for s in segments));run(['-f','concat','-safe','0','-i',str(listing),'-c','copy','-movflags','+faststart',str(v/'funding_demo.mp4')]);(v/'funding_demo.json').write_text(json.dumps(dict(segments=details,physical_time_scale=1.,no_state_edits=True,scope='Title cards, concatenation, resizing/padding only. Source videos reconstruct saved physical qpos for rendering; no animation smoothing or trajectory correction.'),indent=2));print(v/'funding_demo.mp4')
