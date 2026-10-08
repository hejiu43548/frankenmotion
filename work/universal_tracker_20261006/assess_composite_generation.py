import sys,json,argparse
from pathlib import Path
import numpy as np
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/universal_tracker_20261006';sys.path.insert(0,str(R/'outputs_amass/franken_eleven_20261003/code'));import transfer as tr
from audit_results import canonical
parser=argparse.ArgumentParser();parser.add_argument('--folder',default='composite_conditioning_probe');args=parser.parse_args();folder=D/args.folder;rows=json.loads((folder/'manifest.json').read_text());result=[]
for row in rows:
 z=np.load(row['path']);p=canonical(z['joints_zup_m']);h=float(z['human_height']);walk=tr.measure(p,'walk',h);wave=tr.measure(p,'wave',h);r=dict(row,human_walk=walk,human_wave=wave);result.append(r);print(row.get('gain',row['mode']),row['seed'],row['command'],row['wave_command'],'walk',round(walk['quantity'],3),walk['event_pass'],'wave',round(wave['quantity'],3),wave['event_pass'],wave.get('full_cycles'))
(folder/'human_metrics.json').write_text(json.dumps(result,indent=2,default=lambda x:x.item()))
