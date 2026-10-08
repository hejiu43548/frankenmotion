"""Reuse training references only; keep held-out table teacher groups excluded."""
from pathlib import Path
import json, hashlib, numpy as np

R = Path('/home/pku/frankenmotion')
D = R / 'outputs_amass/universal_tracker_20261006'
old = R / 'outputs_amass/franken_unified_20261004/training_corpus_augmented'
out = D / 'joint_corpus_v1'
out.mkdir(exist_ok=False)
records = json.loads((old / 'manifest.json').read_text())
assert len(records) == 735
tables = json.loads((R / 'outputs_amass/turn_demo_20261005/training_full_v1/training_manifest.json').read_text())
for i, row in enumerate(tables[:42]):
    records.append(dict(row, task='table_approach', source=f'table_train_{i:03d}', group=i))
keys = ['joint_pos','joint_vel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w']
buffers = {k: [] for k in keys}
for row in records:
    p = Path(row['motion_path'])
    assert 'final' not in p.parts and 'development_validation' not in p.parts
    with np.load(p) as z:
        assert float(z['fps']) == 50
        for k in keys:
            a = z[k].astype(np.float32)
            assert np.isfinite(a).all()
            buffers[k].append(a)
        row['frames'] = len(z['joint_pos'])
    row['motion_sha256'] = hashlib.sha256(p.read_bytes()).hexdigest()
arrays = {k: np.concatenate(v) for k,v in buffers.items()}
np.savez_compressed(out / 'training_motions.npz', fps=50., **arrays)
(out / 'manifest.json').write_text(json.dumps(records, indent=2))
(out / 'clips.json').write_text(json.dumps(dict(ends=np.cumsum([r['frames'] for r in records]).tolist(), records=records), indent=2))
(out / 'audit.json').write_text(json.dumps(dict(clips=len(records),frames=len(arrays['joint_pos']),task_counts={t:sum(r['task']==t for r in records) for t in sorted(set(r['task'] for r in records))},table_groups=list(range(42)),heldout_table_groups=list(range(42,48)),source='Existing training-only references; no new final/reference validation data',array_shapes={k:list(v.shape) for k,v in arrays.items()}), indent=2))
print(out, len(records), len(arrays['joint_pos']), flush=True)
