"""Verify balanced training identities and strict held-out seed separation."""
from pathlib import Path
import json,hashlib,argparse
from collections import Counter
import numpy as np
R=Path('/home/pku/frankenmotion');U=R/'outputs_amass/franken_unified_20261004';p=argparse.ArgumentParser();p.add_argument('--dataset',choices=['training_corpus_v2','training_corpus_augmented'],default='training_corpus_v2');a=p.parse_args();folder=U/a.dataset;expected=735 if a.dataset.endswith('augmented') else 495;read=lambda p:json.loads(p.read_text());records=read(folder/'manifest.json');dev=read(U/'development_validation/manifest.json');meta=read(folder/'clips.json');key=lambda r:(r['task'],r['seed'],r['command_index'],r['source']);assert len(records)==expected and len({key(r) for r in records})==expected;assert len(dev)==110 and not {key(r) for r in records}&{key(r) for r in dev};counts=Counter(r['task'] for r in records);ordinary={k:v for k,v in counts.items() if k not in ['sequence','composition']};assert len(ordinary)==11 and set(ordinary.values())=={45}
if expected==735:assert counts['sequence']==160 and counts['composition']==80
allowed_seeds=set(range(990700,990711))|{92041000+100*pi+si for pi in range(4) for si in range(2)};frames=[]
for r in records:
 z=np.load(r['motion_path']);assert float(z['fps'])==50 and all(np.isfinite(z[k]).all() for k in z.files);assert len(z['joint_pos'])==r['frames'] and z['joint_pos'].shape[1]==29
 assert hashlib.sha256(Path(r['reference_path']).read_bytes()).hexdigest()==r['reference_sha256'];assert r['seed'] in allowed_seeds and '/development_validation/' not in r['reference_path'] and '/final_test/' not in r['reference_path'];frames.append(r['frames'])
assert np.array_equal(np.cumsum(frames),meta['ends']);z=np.load(folder/'training_motions.npz');assert len(z['joint_pos'])==sum(frames)
report=dict(training_clips=expected,development_validation_requests=110,task_clip_counts=dict(counts),training_validation_identity_overlap=0,final_seed_range_absent=True,total_frames=sum(frames),all_source_reference_hashes_verified=True,all_motion_arrays_finite=True)
(folder/'audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
