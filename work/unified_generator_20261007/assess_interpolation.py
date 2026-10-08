import sys,json
from pathlib import Path
import numpy as np
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/unified_generator_20261007';sys.path.insert(0,str(R/'outputs_amass/franken_eleven_20261003/code'))
from audit_results import native_metrics,TASKS,RANGES,TOLS
out=D/'interpolation_probe';rows=json.loads((out/'manifest.json').read_text())
for r in rows:
 m=native_metrics(r);tid=TASKS.index(r['task']);r['human']=m;r['joint_pass']=bool(m['event_pass'] and abs(m['quantity']-r['command'])<=TOLS[tid]);r['semantic_E']=min(abs(m['quantity']-r['command'])/(RANGES[tid][1]-RANGES[tid][0]),1) if m['event_pass'] else 1
s={v:dict(count=sum(r['variant']==v for r in rows),joint_pass=sum(r['joint_pass'] for r in rows if r['variant']==v),event_pass=sum(r['human']['event_pass'] for r in rows if r['variant']==v),semantic_E=float(np.mean([r['semantic_E'] for r in rows if r['variant']==v]))) for v in ['old','unified']}
(out/'audit.json').write_text(json.dumps(s,indent=2,default=lambda x:x.item()));(out/'audited_results.json').write_text(json.dumps(rows,indent=2,default=lambda x:x.item()));print(json.dumps(s,default=lambda x:x.item()))
