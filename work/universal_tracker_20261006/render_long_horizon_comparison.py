"""Full synchronized physical rollouts: reference anchoring versus fixed world reference."""
from pathlib import Path
import json,subprocess,hashlib
import imageio_ffmpeg
from PIL import Image,ImageDraw,ImageFont
D=Path('/home/pku/frankenmotion/outputs_amass/universal_tracker_20261006');V=D/'visuals/frozen_long_horizon';sources=[V/'four_cycles_follow.mp4',V/'four_cycles_fixed_world.mp4'];font='/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf';a=json.loads((D/'long_horizon_ablation.json').read_text());header=Image.new('RGB',(1920,80),'#101b2a');draw=ImageDraw.Draw(header)
for i,(key,title) in enumerate([('relative_anchors','Relative reference anchors'),('fixed_world_reference','Fixed world reference')]):
 error=a['results'][key]['initial_world_plan_final_root_xy_error_m'];subtitle=f'End deviation from INITIAL world plan = {error:.3f} m';draw.text((960*i+24,12),title,font=ImageFont.truetype(font,26),fill='white');draw.text((960*i+24,48),subtitle,font=ImageFont.truetype(font,19),fill='#9fceca')
header_path=V/'comparison_header.png';header.save(header_path);frames=[json.loads(p.with_suffix('.json').read_text())['frames'] for p in sources];assert frames[0]==frames[1]
filters='[0:v]scale=960:540[v0];[1:v]scale=960:540[v1];[v0][v1]hstack=inputs=2[body];[2:v][body]vstack=inputs=2[out]';out=V/'long_horizon_comparison.mp4';subprocess.run([imageio_ffmpeg.get_ffmpeg_exe(),'-v','error','-y','-i',str(sources[0]),'-i',str(sources[1]),'-loop','1','-framerate','25','-i',str(header_path),'-filter_complex',filters,'-map','[out]','-frames:v',str(frames[0]),'-r','25','-c:v','libx264','-crf','20','-preset','fast','-pix_fmt','yuv420p',str(out)],check=True)
(out.with_suffix('.json')).write_text(json.dumps(dict(scope=__doc__,sources=[dict(path=str(p),sha256=hashlib.sha256(p.read_bytes()).hexdigest()) for p in sources],full_clips=True,time_scale=1.,visual_edits='Aspect-preserving downscale, side-by-side composition, explicit mode labels and final deviation from initial plan. No temporal edits or state changes.',experiment=a),indent=2));print(out)
