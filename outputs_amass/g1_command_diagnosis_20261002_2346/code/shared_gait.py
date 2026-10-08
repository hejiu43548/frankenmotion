"""Fourth intervention: shared calibrated G1 legs + each FrankenMotion upper body."""
import os,json
from pathlib import Path
import numpy as np
import run,periodic,g1_runtime as rt
OUT=Path(os.environ['DIAG_OUT'])
run.SOURCES=OUT/'variant_baselines/retarget'
mapping=json.loads((OUT/'frozen_command_map.json').read_text())['walk']
template='01_walk_speed_v1';f,coef=periodic.fit(template)
cases=['01_walk_speed_v1','01_walk_speed_v2','01_walk_speed_v3','04_walk_wave_v1','04_walk_wave_v2','04_walk_wave_v3']
commands=[.25,.35,.45,.55];seeds=[6613,7724]
run.write('shared_gait_protocol.json',dict(cases=cases,commands=commands,seeds=seeds,template=template,
    note='Deliberate lower-body replacement with a shared calibrated G1 gait, not faithful tracking of generated legs. All 17 upper-body joints remain source-specific and unretimed. Heading target zero. No calibration on these new perturbation outcomes.'))
for case in cases:periodic.FITS[case]=(f,coef)
run.reference=periodic.reference;m=rt.load_model();p=rt.Policy();results=[]
for case in cases:
    for cmd in commands:
        cadence=float(np.interp(cmd,mapping['speeds'],mapping['cadences']))
        for seed in seeds:
            label=f'shared_{case}_v{cmd:.2f}_s{seed}'
            r=run.evaluate(m,p,case,cadence,1.6,label,seed)
            r.update(command_mps=cmd,method='shared_robot_gait',calibration_source=template,
                     source_heldout=case!=template,
                     success=bool(r['complete'] and abs(r['speed_mps']-cmd)<=.05 and r['max_heading_drift_deg']<15))
            results.append(r);run.write('shared_gait_results.json',results)
run.write('shared_gait_status.json',dict(stage='complete',count=len(results)))
