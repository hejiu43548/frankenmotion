"""Frozen command map, interpolation commands, and uncalibrated source variants."""
import json,os
import numpy as np
from pathlib import Path
import run,periodic,g1_runtime as rt
OUT=Path(os.environ['DIAG_OUT'])
rows=json.loads((OUT/'periodic_sweep.json').read_text())
maps={}
for family,case in [('walk','01_walk_speed_v1'),('wave','04_walk_wave_v2')]:
    selected=sorted([r for r in rows if r['case']==case and r['amplitude']==1.6],key=lambda x:x['cadence'])
    assert all(r['complete'] and r['max_heading_drift_deg']<15 for r in selected)
    speed=np.array([r['speed_mps'] for r in selected]);assert np.all(np.diff(speed)>0)
    maps[family]=dict(source=case,speeds=speed.tolist(),cadences=[r['cadence'] for r in selected],amplitude=1.6)
run.write('frozen_command_map.json',maps)
run.SOURCES=OUT/'variant_baselines/retarget';run.reference=periodic.reference
cases=['01_walk_speed_v1','01_walk_speed_v2','01_walk_speed_v3','04_walk_wave_v1','04_walk_wave_v2','04_walk_wave_v3']
commands=[.25,.35,.45,.55];seeds=[4411,5522]
run.write('transfer_protocol.json',dict(cases=cases,commands=commands,seeds=seeds,amplitude=1.6,
    note='Command mapping frozen from two calibration clips. New command values and perturbation seeds. Four source variants were not used for response calibration but share text/noise with parent source; not independent random source families.'))
m=rt.load_model();p=rt.Policy();result=[]
for case in cases:
    family='walk' if case.startswith('01') else 'wave';mapping=maps[family]
    for cmd in commands:
        cadence=float(np.interp(cmd,mapping['speeds'],mapping['cadences']))
        for seed in seeds:
            label=f'transfer_{case}_v{cmd:.2f}_s{seed}'
            r=run.evaluate(m,p,case,cadence,1.6,label,seed)
            r.update(command_mps=cmd,method='frozen_map_periodic_gait',calibration_source=mapping['source'],
                     source_heldout=case!=mapping['source'],
                     success=bool(r['complete'] and abs(r['speed_mps']-cmd)<=.05 and r['max_heading_drift_deg']<15))
            result.append(r);run.write('transfer_results.json',result)
run.write('transfer_status.json',dict(stage='complete',count=len(result)))
