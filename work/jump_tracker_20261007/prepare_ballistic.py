"""Training-only heuristic COM flight repair; same non-flight-task corpus reused.
This is not a contact/torque-feasibility optimizer. Preserve that limitation.
"""
import sys,json,hashlib
from pathlib import Path
import numpy as np
R=Path('/home/pku/frankenmotion');W=R/'work/jump_tracker_20261007';U=R/'outputs_amass/universal_tracker_20261006';D=R/'outputs_amass/jump_tracker_20261007';sys.path[:0]=[str(W),str(R/'work/universal_tracker_20261006')]
from ballistic_repair import repair
from fk_conversion import convert
out=D/'ballistic_corpus';out.mkdir(exist_ok=False);(out/'motions').mkdir();(out/'references').mkdir();rows=json.loads((U/'joint_corpus_expanded_corrected/manifest.json').read_text());records=[];buffers={k:[] for k in ['joint_pos','joint_vel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w']}
for index,row in enumerate(rows):
 r=dict(row)
 if row['task'] in ['jump','extra_jump']:
  q=np.load(row['reference_path'])['reference_qpos'];q,notes=repair(q);name=f'{index:04d}_'+Path(row['motion_path']).stem;rp=out/'references'/(name+'.npz');mp=out/'motions'/(name+'.npz');np.savez_compressed(rp,reference_qpos=q);arrays=convert(q,entry=row['task']=='jump');np.savez_compressed(mp,fps=50.,**arrays);r.update(original_reference_path=row['reference_path'],reference_path=str(rp),motion_path=str(mp),frames=len(arrays['joint_pos']),reference_sha256=hashlib.sha256(rp.read_bytes()).hexdigest(),motion_sha256=hashlib.sha256(mp.read_bytes()).hexdigest(),ballistic_repair=notes)
 else:
  z=np.load(row['motion_path']);arrays={k:z[k] for k in buffers}
 for k in buffers:buffers[k].append(arrays[k].astype(np.float32))
 records.append(r)
 if len(records)%100==0:print(len(records),flush=True)
np.savez_compressed(out/'training_motions.npz',fps=50.,**{k:np.concatenate(v) for k,v in buffers.items()});(out/'manifest.json').write_text(json.dumps(records,indent=2));(out/'clips.json').write_text(json.dumps(dict(ends=np.cumsum([r['frames'] for r in records]).tolist(),records=records),indent=2));(out/'protocol.json').write_text(json.dumps(dict(scope=__doc__,clips=len(records),modified=sum('ballistic_repair' in r for r in records),repaired_flights=sum(len(r.get('ballistic_repair',[])) for r in records),original_dataset='joint_corpus_expanded_corrected'),indent=2))
# Development probe references are disjoint from training corpus.
dev=[]
for row in json.loads((U/'ballistic_reference_probe/manifest.json').read_text()):
 if row['task']=='jump':dev.append(row)
(D/'dev_ballistic_manifest.json').write_text(json.dumps(dev,indent=2))
