import sys,os,json
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/home/pku/frankenmotion');OUT=Path(os.environ['DIAG_OUT'])
sys.path[:0]=[str(ROOT),str(ROOT/'work/franken_sim_bridge')];os.chdir(ROOT)
from src.tools.inference import load_smplh
from bridge import decode_fk,export_source
torch.set_num_threads(2);smpl=load_smplh();rows=[]
folder=ROOT/'outputs_amass/root_control_20260925/parameter_videos_6'
manifest=json.loads((folder/'manifest.json').read_text())
for case in manifest:
    if case['id'] not in ['01_walk_speed','02_jog_speed','04_walk_wave']:continue
    for i,variant in enumerate(case['variants'],1):
        name=case['id']+f'_v{i}';path=OUT/'human_variants'/(name+'.npz')
        with torch.no_grad():
            decoded=decode_fk(torch.from_numpy(np.load(folder/'motions'/(name+'.npz'))['motion']).float(),smpl)
            if not path.exists():export_source(decoded,path)
        j=decoded['joints_zup_m'].numpy();left=j[:,1]-j[:,2];up=j[:,12]-j[:,0];forward=np.cross(left,up);forward/=np.linalg.norm(forward,axis=1,keepdims=True)
        duration=(len(j)-1)/20
        speed=float((j[-1,0]-j[0,0])@forward[0]/duration)
        rows.append(dict(case=name,requested=variant,human_fixed_heading_speed_mps=speed,seed=case['seed'],duration_s=duration))
        print(rows[-1],flush=True)
(OUT/'human_variant_metrics.json').write_text(json.dumps(rows,indent=2))
