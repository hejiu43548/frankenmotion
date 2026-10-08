"""Same-source, same-command, same-initial-perturbation comparison to transfer."""
import os,json
from pathlib import Path
import numpy as np
import run,periodic,g1_runtime as rt
BASE=Path(os.environ['DIAG_OUT']);OUT=BASE/'paired_check';OUT.mkdir(exist_ok=False);(OUT/'rollouts').mkdir()
run.OUT=OUT;run.SOURCES=BASE/'variant_baselines/retarget';run.reference=periodic.reference
mapping=json.loads((BASE/'frozen_command_map.json').read_text())['walk']
fit=json.loads((BASE/'periodic_fit/01_walk_speed_v1.json').read_text())
cases=json.loads((BASE/'transfer_results.json').read_text());results=[]
for r in cases:periodic.FITS[r['case']]=(fit['frequency_hz'],np.array(fit['coefficients']))
m=rt.load_model();p=rt.Policy()
for old in cases:
    cmd=old['command_mps'];case=old['case'];seed=old['seed'];cadence=float(np.interp(cmd,mapping['speeds'],mapping['cadences']))
    r=run.evaluate(m,p,case,cadence,1.6,f'paired_{case}_v{cmd:.2f}_s{seed}',seed)
    r.update(command_mps=cmd,previous_success=old['success'],previous_speed_mps=old['speed_mps'],
             success=bool(r['complete'] and abs(r['speed_mps']-cmd)<=.05 and r['max_heading_drift_deg']<15))
    results.append(r);run.write('results.json',results)
run.write('status.json',dict(status='complete',count=len(results)))
