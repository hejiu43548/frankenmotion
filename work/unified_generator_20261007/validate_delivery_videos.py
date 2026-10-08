from pathlib import Path
import json,imageio_ffmpeg
from concurrent.futures import ThreadPoolExecutor
D=Path('/home/pku/frankenmotion/outputs_amass/unified_generator_20261007');files=list((D/'visuals/unified_final').glob('*.mp4'))+list((D/'visuals/g1_unified_final').glob('*/*.mp4'));assert len(files)==44
frames={'raise_hand':60,'reach':60,'strike':60,'wave':120,'turn':120,'sidestep':120,'back_walk':120,'kick':60,'jump':60,'lean':60,'walk':120}
def check(p):
 it=imageio_ffmpeg.read_frames(str(p));meta=next(it);count=sum(1 for _ in it)
 if p.parent.name=='unified_final':expected=frames[p.stem]*3;fps=20
 else:expected=json.loads((p.parent/'render_protocol.json').read_text())['frames'];fps=25
 assert count==expected,(str(p),count,expected);assert abs(meta['fps']-fps)<.001
 return dict(path=str(p),frames=count,fps=meta['fps'],size=meta['size'])
with ThreadPoolExecutor(4) as pool:rows=list(pool.map(check,files))
(D/'report/video_validation.json').write_text(json.dumps(dict(videos=len(rows),all_frames_decoded=True,rows=rows),indent=2));print('validated',len(rows),sum(r['frames'] for r in rows))
