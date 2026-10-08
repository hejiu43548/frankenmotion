import os,sys,json,hashlib
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/home/pku/frankenmotion');OUT=Path(os.environ['FULL_OUT'])
sys.path[:0]=[str(ROOT),str(ROOT/'work/franken_sim_bridge')];os.chdir(ROOT)
from src.tools.inference import load_smplh
from bridge import decode_fk,export_source
torch.set_num_threads(2);smpl=load_smplh()
folder=ROOT/'outputs_amass/root_control_20260925/parameter_videos_6'
manifest=json.loads((folder/'manifest.json').read_text());rows=[]
for family in manifest:
    for i,variant in enumerate(family['variants'],1):
        case=family['id']+f'_v{i}';path=OUT/'human'/(case+'.npz')
        with torch.no_grad():
            if not path.exists():export_source(decode_fk(torch.from_numpy(np.load(folder/'motions'/(case+'.npz'))['motion']).float(),smpl),path)
        rows.append(dict(case=case,requested=variant,seed=family['seed'],source_file=str(folder/'motions'/(case+'.npz'))))
(OUT/'sources.json').write_text(json.dumps(rows,indent=2))
asset=OUT/'human_joints_info.pkl'
assert hashlib.sha256(asset.read_bytes()).hexdigest()=='4de0bae69caf31e8829a2d3e8adecd887f29115af60a0b8d59237dfbfea1c975'
data=torch.load(asset,map_location='cpu',weights_only=False)
np.savez(OUT/'official_human_skeleton.npz',J=np.asarray(data['J']),parents=np.asarray(data['parents_list']))
print('Exported',len(rows),'source clips; official skeleton hash verified.')
