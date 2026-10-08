"""Joint training corpus with source-disjoint official mocap categories.
New references keep original motion; artificial standing entry is removed for RSI
training. Existing 11-task and table references are byte-for-byte reused.
"""
import json,hashlib,argparse
from pathlib import Path
import numpy as np
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/universal_tracker_20261006';p=argparse.ArgumentParser();p.add_argument('--name',default='joint_corpus_expanded');p.add_argument('--natural',nargs='+',default=['natural_extended_motion/manifest.json']);a=p.parse_args();out=D/a.name;out.mkdir(exist_ok=False);(out/'extra_motions').mkdir()
records=json.loads((D/'joint_corpus_v1/manifest.json').read_text());natural=[row for name in a.natural for row in json.loads((D/name).read_text())];train=[r for r in natural if r['split']=='train'];val_sources={r['source'] for r in natural if r['split']=='val'};assert all(r['source'] not in val_sources for r in train)
keys=['joint_pos','joint_vel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w']
for row in train:
 with np.load(row['motion_path']) as z:arrays={k:z[k][50:].astype(np.float32) for k in keys}
 assert len(arrays['joint_pos'])>=50
 p=out/'extra_motions'/(Path(row['path']).stem+'_rsi_motion.npz');np.savez_compressed(p,fps=50.,**arrays)
 records.append(dict(row,original_task=row['task'],task='extra_'+row['task'],motion_path=str(p),frames=len(arrays['joint_pos']),motion_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),entry='original reference, no artificial standing prefix'))
buffers={k:[] for k in keys}
for row in records:
 with np.load(row['motion_path']) as z:
  for k in keys:assert np.isfinite(z[k]).all();buffers[k].append(z[k].astype(np.float32))
arrays={k:np.concatenate(v) for k,v in buffers.items()};np.savez_compressed(out/'training_motions.npz',fps=50.,**arrays);(out/'manifest.json').write_text(json.dumps(records,indent=2));(out/'clips.json').write_text(json.dumps(dict(ends=np.cumsum([r['frames'] for r in records]).tolist(),records=records),indent=2));(out/'audit.json').write_text(json.dumps(dict(clips=len(records),new_source_count=len(train),frames=len(arrays['joint_pos']),tasks={task:sum(r['task']==task for r in records) for task in sorted({r['task'] for r in records})},source_overlap_with_natural_val=0,scope='Exploratory joint training while the original 11-task criterion remains unmet; not a claim of generality. No val/test trajectories are in this corpus.'),indent=2));print(len(records),len(train),len(arrays['joint_pos']),flush=True)
