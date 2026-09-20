import json,csv
from pathlib import Path
from collections import Counter
import imageio.v2 as imageio
r=Path('/home/pku/frankenmotion/outputs_amass/subset_review_200');m=json.loads((r/'manifest.json').read_text());assert len(m['items'])==200;assert len({x['id'] for x in m['items']})==200
assert Counter(x['primary'] for x in m['items'])=={'walk':50,'run':50,'wave':50,'throw':50}
results=[]
for x in m['items']:
 p=r/x['video'];reader=imageio.get_reader(p);meta=reader.get_meta_data();n=reader.count_frames();assert n==x['end_frame']-x['start_frame'],x['id'];assert meta['fps']==20;reader.close();assert (r/x['labels']).exists();results.append({'id':x['id'],'frames':n,'bytes':p.stat().st_size})
(r/'verification.json').write_text(json.dumps({'passed':True,'videos':200,'counts':dict(Counter(x['primary'] for x in m['items'])),'frame_counts_and_fps_verified':True,'results':results},indent=2));print('VERIFIED 200 videos, exact frame counts, 20 FPS, labels matched')
