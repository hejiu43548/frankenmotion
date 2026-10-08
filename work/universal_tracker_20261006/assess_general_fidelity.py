"""Reference tracking accuracy, intentionally not a semantic task success claim."""
import argparse,json
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();root=Path(a.run);assert not (root/'INVALIDATED.json').exists();rows=[]
for row in json.loads((root/'results.json').read_text()):
 r=dict(row);r['accurate_complete']=False
 if 'error' not in row:
  d=Path(row['run']);actual=np.load(d/'actual.npz');ref=np.load(d/'motion.npz');n=len(actual['body_pos_w']);start=min(50 if row.get('entry','standing')=='standing' else 0,n-1);per_frame=np.linalg.norm(actual['body_pos_w'][start:]-ref['body_pos_w'][start:n],axis=-1).mean(-1)
  r['global_mpjpe_recomputed_m']=float(per_frame.mean());r['frame_mpjpe_p95_m']=float(np.quantile(per_frame,.95));assert abs(r['global_mpjpe_recomputed_m']-r['global_mpjpe_m'])<1e-6
  r['accurate_complete']=bool(row['physical_complete'] and per_frame.mean()<=.10 and np.quantile(per_frame,.95)<=.25)
 rows.append(r)
summary={}
for task in sorted({r['task'] for r in rows}):
 rr=[r for r in rows if r['task']==task];summary[task]=dict(requests=len(rr),complete=sum(r['physical_complete'] for r in rr),accurate_complete=sum(r['accurate_complete'] for r in rr),mean_error_completed_m=float(np.mean([r['global_mpjpe_m'] for r in rr if r['physical_complete']])) if any(r['physical_complete'] for r in rr) else None)
 completed=[r for r in rr if r['physical_complete']]
 summary[task]['mean_root_aligned_error_completed_m']=float(np.mean([r['root_aligned_mpjpe_m'] for r in completed])) if completed else None
 summary[task]['mean_root_translation_error_completed_m']=float(np.mean([r['root_translation_error_m'] for r in completed])) if completed else None
(root/'fidelity_results.json').write_text(json.dumps(rows,indent=2));(root/'fidelity_summary.json').write_text(json.dumps(dict(scope='Completion plus global mean body-position error <=0.10m and frame-wise mean error P95<=0.25m. Not semantic success, not proof of naturalness; failures remain in denominator.',results=summary),indent=2));print(json.dumps(summary,indent=2))
