"""Prepare source-disjoint motion candidates for exploratory generalization.

These are mocap references, not FrankenMotion-generated demonstrations and not
newly learned numeric commands. Only train/val sources are loaded here.
"""
import os,sys,json,hashlib,argparse
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','1')
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/universal_tracker_20261006'
sys.path.insert(0,str(R/'outputs_amass/franken_eleven_20261003/code'))
import numpy as np,torch
from core import FK
torch.set_num_threads(2)
parser=argparse.ArgumentParser();parser.add_argument('--name',default='natural_candidates');parser.add_argument('--inventory',default='action_inventory.json');parser.add_argument('--train-limit',type=int,default=6);parser.add_argument('--val-limit',type=int,default=2);parser.add_argument('--max-frames',type=int,default=120);args=parser.parse_args()
inventory=json.loads((D/args.inventory).read_text())
out=D/args.name;out.mkdir(exist_ok=False);fk=FK('cpu');manifest=[]
for split,limit in [('train',args.train_limit),('val',args.val_limit)]:
    seen=set()
    for category,items in inventory['candidate_manifest'][split].items():
        for row in items[:limit]:
            if row['source'] in seen:continue
            seen.add(row['source']);p=Path(row['motion_path']);raw=np.load(p)
            start=int(round(row['start']*20));end=min(int(round(row['end']*20)),start+args.max_frames)
            raw=raw[start:end]
            assert raw.ndim==2 and raw.shape[1]==205 and len(raw)>=20,(p,raw.shape,start,end)
            with torch.no_grad():positions,poses,root=fk(torch.tensor(raw[None],dtype=torch.float32),canonical=False,return_pose=True)
            dest=out/f'{split}_{category}_{row["uid"]}.npz'
            np.savez_compressed(dest,motion=raw,joints_zup_m=positions[0].numpy(),poses_axisangle=poses[0].numpy(),root_translation=root[0].numpy(),human_height=fk.height,fps=20.)
            manifest.append(dict(row,split=split,task=category,path=str(dest),frames=len(raw),source_type='official_annotated_mocap',source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),human_sha256=hashlib.sha256(dest.read_bytes()).hexdigest()))
(out/'manifest.json').write_text(json.dumps(manifest,indent=2,ensure_ascii=False))
print('Prepared',len(manifest),'mocap candidates; no final test motion loaded',flush=True)
