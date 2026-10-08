"""Source-disjoint mocap tracking test after global checkpoint freeze.
These are source motions, not generated demonstrations or unseen-generator data.
"""
import os,sys,json,hashlib
from pathlib import Path
os.environ.setdefault('OMP_NUM_THREADS','1')
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/universal_tracker_20261006';sys.path.insert(0,str(R/'outputs_amass/franken_eleven_20261003/code'))
import numpy as np,torch
from core import FK
selection=D/'frozen_unified/protocol.json';chosen=json.loads(selection.read_text());assert hashlib.sha256(Path(chosen['checkpoint']).read_bytes()).hexdigest()==chosen['checkpoint_sha256'];inventory=json.loads((D/'action_inventory_v2.json').read_text());forbidden=set()
for name in ['natural_extended_corrected_motion','natural_locomotion_motion']:
 path=D/name/'manifest.json'
 assert path.exists(),path
 forbidden.update(r['source'] for r in json.loads(path.read_text()))
out=D/'natural_test';out.mkdir(exist_ok=False);torch.set_num_threads(4);fk=FK('cpu');rows=[];seen=set()
for category,items in inventory['candidate_manifest']['test'].items():
 for row in items[:5]:
  assert row['source'] not in forbidden
  if row['source'] in seen:continue
  seen.add(row['source']);p=Path(row['motion_path']);raw=np.load(p);start=int(round(row['start']*20));end=min(int(round(row['end']*20)),start+300);raw=raw[start:end];assert raw.ndim==2 and raw.shape[1]==205 and len(raw)>=20
  with torch.no_grad():pos,poses,root=fk(torch.tensor(raw[None],dtype=torch.float32),canonical=False,return_pose=True)
  dest=out/f'test_{category}_{row["uid"]}.npz';np.savez_compressed(dest,motion=raw,joints_zup_m=pos[0].numpy(),poses_axisangle=poses[0].numpy(),root_translation=root[0].numpy(),human_height=fk.height,fps=20.);rows.append(dict(row,split='test',task=category,path=str(dest),frames=len(raw),source_type='official_annotated_mocap',source_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),human_sha256=hashlib.sha256(dest.read_bytes()).hexdigest()));print(category,row['uid'],flush=True)
(out/'manifest.json').write_text(json.dumps(rows,indent=2));(out/'protocol.json').write_text(json.dumps(dict(scope=__doc__,selection_sha256=hashlib.sha256(selection.read_bytes()).hexdigest(),inventory_sha256=hashlib.sha256((D/'action_inventory_v2.json').read_bytes()).hexdigest(),limit_per_category=5,max_frames=300,unique_sources=len(seen),training_validation_overlap=len(seen&forbidden),sampling='Deterministic inventory first5, cross-category source deduplication; no tracker-based filtering.',motion_pretraining_caveat='Source isolation established for this tracker experiment; not proof that pretrained generator/foundation controllers never saw related data.'),indent=2))
