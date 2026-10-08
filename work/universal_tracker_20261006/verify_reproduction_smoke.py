"""Re-execute the deterministic first frozen reference and compare actual states."""
from pathlib import Path
import json,numpy as np
D=Path('/home/pku/frankenmotion/outputs_amass/universal_tracker_20261006');now=json.loads((D/'general_evaluation/reproduction_smoke/results.json').read_text())[0];old=next(r for r in json.loads((D/'general_evaluation/fresh_candidate/results.json').read_text()) if r['motion_path']==now['motion_path']);a=np.load(Path(old['run'])/'actual.npz');b=np.load(Path(now['run'])/'actual.npz');result={}
for k in ['qpos','qvel','body_pos_w']:
 assert a[k].shape==b[k].shape;error=float(np.max(abs(a[k]-b[k])));assert error==0,(k,error);result[k]=dict(shape=list(a[k].shape),max_absolute_error=error)
p=D/'release/reproduction_smoke.json';p.write_text(json.dumps(dict(scope=__doc__,source=old['source'],command=old['command'],original_run=old['run'],repeated_run=now['run'],physical_complete=now['physical_complete'],arrays=result,limitations='One CPU rollout; not cross-hardware determinism or reproduction of training.'),indent=2));print(result)
