import json,hashlib
from pathlib import Path
import numpy as np,torch
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/table_demo_20261005';f=json.loads((D/'frozen/protocol.json').read_text());s=json.loads((D/'final_test/unified_summary.json').read_text());assert len(s['results'])==32;torch.set_num_threads(2);actor=torch.jit.load(str(D/'frozen/actor.pt'),map_location='cpu').eval();reports=[]
with torch.inference_mode():
 for row in s['results']:
  run=Path(row['scene']['source'])/'unified';assert row['checkpoint_sha256']==f['tracker_sha256'];z=np.load(run/'actual.npz');prediction=actor(torch.from_numpy(z['observations'])).numpy();error=float(abs(prediction-z['actions']).max());assert error<2e-4,(run,error);reports.append(dict(scene_index=row['scene']['index'],frames=len(prediction),action_max_error=error))
source=[]
for name,expected in f['source_sha256'].items():
 path=R/'work'/name;actual=hashlib.sha256(path.read_bytes()).hexdigest();source.append(dict(name=name,unchanged=actual==expected,sha256=actual));assert actual==expected,('Frozen source changed',name)
(D/'frozen/parity_audit.json').write_text(json.dumps(dict(tracker_sha256=f['tracker_sha256'],actor_sha256=hashlib.sha256((D/'frozen/actor.pt').read_bytes()).hexdigest(),cases=reports,all_final_actions_match_exported_actor=True,max_action_error=max(r['action_max_error'] for r in reports),frozen_sources=source),indent=2));print('PARITY PASS',len(reports),sum(x['frames'] for x in reports),max(x['action_max_error'] for x in reports))
