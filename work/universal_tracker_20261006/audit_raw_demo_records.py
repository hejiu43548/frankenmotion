"""Structural checks on archived actual-state recordings; not a standalone proof of simulation authenticity."""
from pathlib import Path
import json,hashlib
import numpy as np
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'outputs/universal_tracker_20261006';files=[D/p for p in (D/'demo_evidence_files.txt').read_text().splitlines() if p.endswith('/actual.npz')];rows=[]
for p in files:
 z=np.load(p);q=z['qpos'];v=z['qvel'];assert np.isfinite(q).all() and np.isfinite(v).all();assert len(q)==len(v);normerr=float(np.max(np.abs(np.linalg.norm(q[:,3:7],axis=-1)-1)));assert normerr<1e-5;rootstep=np.linalg.norm(np.diff(q[:,:3],axis=0),axis=-1);result=json.loads((p.parent/'result.json').read_text());anchors=p.parent/'anchor_events.json';events=json.loads(anchors.read_text()) if anchors.exists() else []
 rows.append(dict(run=str(p.parent),raw_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),frames=len(q),finite=True,quaternion_max_norm_error=normerr,root_step_max_m=float(rootstep.max()),root_step_p99_m=float(np.quantile(rootstep,.99)),physical_complete=result['physical_complete'],anchor_event_count=len(events),scope='Actual state archive, not reference poses. Reference anchors are recorded separately; no interpretation of these structural checks as semantic success.'))
(D/'raw_demo_integrity.json').write_text(json.dumps(dict(scope=__doc__,records=rows),indent=2));print('Verified',len(rows),'finite actual-state recordings; largest root step',max(r['root_step_max_m'] for r in rows))
