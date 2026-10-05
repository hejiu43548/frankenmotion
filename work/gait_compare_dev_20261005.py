import json,subprocess
from pathlib import Path
import numpy as np
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/gait_demo_20261005';O=R/'outputs_amass/table_demo_20261005';rows=[]
for i in range(6):
 for label,run in [('previous',O/f'development_v3/scene_{i:03d}/selected_fresh'),('retime_only',D/f'dev_timed/scene_{i:03d}/baseline_timed'),('new_single_actor',D/f'dev_timed/scene_{i:03d}/distill_v2_3000_gpu')]:
  subprocess.run([str(R/'work/mjlab_stable_env/bin/python'),str(R/'work/gait_metrics_20261005.py'),'--run',str(run)],check=True,stdout=subprocess.DEVNULL)
  rows.append(dict(condition=label,run=str(run),**json.loads((run/'gait_metrics.json').read_text())))
out=dict(rows=rows,summary={})
for label in ['previous','retime_only','new_single_actor']:
 rr=[r for r in rows if r['condition']==label];out['summary'][label]={k:float(np.mean([r[k] for r in rr])) for k in ['knee_std_deg','knee_amplitude_ratio','clearance_p95_m','double_support_fraction','flight_fraction','mean_contact_point_slip_m_s','root_error_m','palm_error_m']};out['summary'][label]['successes']=sum(r['success'] for r in rr)
(D/'development_comparison.json').write_text(json.dumps(out,indent=2));print(json.dumps(out['summary'],indent=2))
