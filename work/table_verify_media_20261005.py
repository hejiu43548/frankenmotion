import json,hashlib
from pathlib import Path
import imageio.v2 as imageio
D=Path('/home/pku/frankenmotion/outputs_amass/table_demo_20261005');files=list((D/'videos').glob('*.mp4'))+[D/'command_sweep/direction_distance_demo.mp4',D/'cpu_final/presentation.mp4'];rows=[]
for p in files:
 reader=imageio.get_reader(p);meta=reader.get_meta_data();n=0
 for frame in reader:assert frame.ndim==3 and frame.shape[2]==3;n+=1
 reader.close();expected=1124 if p.name=='funding_demo.mp4' else 437;assert n==expected,(p,n,expected);assert meta['fps']==25;rows.append(dict(file=str(p),frames=n,fps=meta['fps'],duration_s=n/25,size=meta['size'],bytes=p.stat().st_size,sha256=hashlib.sha256(p.read_bytes()).hexdigest()))
(D/'media_audit.json').write_text(json.dumps(dict(all_frames_decoded=True,files=rows),indent=2));print('ALL VIDEOS DECODED',[(Path(r['file']).name,r['frames']) for r in rows])
