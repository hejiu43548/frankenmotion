"""Run one bounded, validated-range simulation with shared G1 legs."""
import argparse,json,os
from pathlib import Path
import numpy as np
import run,periodic,g1_runtime as rt
ap=argparse.ArgumentParser()
ap.add_argument('--case',default='04_walk_wave_v3',choices=['01_walk_speed_v1','01_walk_speed_v2','01_walk_speed_v3','04_walk_wave_v1','04_walk_wave_v2','04_walk_wave_v3'])
ap.add_argument('--speed',type=float,required=True)
ap.add_argument('--seed',type=int,default=6613)
ap.add_argument('--run-id',required=True)
a=ap.parse_args()
if not .25<=a.speed<=.55:ap.error('Only 0.25–0.55 m/s was validated; refusing silent saturation/extrapolation.')
if not a.run_id.replace('-','').replace('_','').isalnum():ap.error('run-id must be alphanumeric, hyphens or underscores')
BASE=Path(os.environ['DIAG_OUT']);destination=BASE/'demos'/a.run_id
destination.mkdir(parents=True,exist_ok=False);(destination/'rollouts').mkdir()
mapping=json.loads((BASE/'frozen_command_map.json').read_text())['walk']
fit=json.loads((BASE/'periodic_fit/01_walk_speed_v1.json').read_text())
run.OUT=destination;run.SOURCES=BASE/'variant_baselines/retarget'
periodic.FITS[a.case]=(fit['frequency_hz'],np.array(fit['coefficients']))
run.reference=periodic.reference
cadence=float(np.interp(a.speed,mapping['speeds'],mapping['cadences']))
r=run.evaluate(rt.load_model(),rt.Policy(),a.case,cadence,1.6,'demo',a.seed)
r.update(command_mps=a.speed,method='shared_robot_gait',source_leg_motion_replaced=True,
         success=bool(r['complete'] and abs(r['speed_mps']-a.speed)<=.05 and r['max_heading_drift_deg']<15))
run.write('result.json',r)
print(json.dumps(r,indent=2))
