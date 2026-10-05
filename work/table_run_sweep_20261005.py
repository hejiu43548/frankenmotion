"""Systematic post-freeze demos; identical generation noise within each row."""
import subprocess,json,math,hashlib
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/table_demo_20261005';f=json.loads((D/'frozen/protocol.json').read_text());out=D/'command_sweep';out.mkdir(exist_ok=False)
cases=[]
for i,angle in enumerate([-.5,0.,.5]):cases.append(dict(name=f'sweep_direction_{i}',group='direction',distance=1.25,angle=angle,seed=58007000))
for i,distance in enumerate([.85,1.3,1.75]):cases.append(dict(name=f'sweep_distance_{i}',group='distance',distance=distance,angle=0.,seed=58008000))
(out/'planned.json').write_text(json.dumps(cases,indent=2));results=[]
for case in cases:
 radius=case['distance']+.65;x=radius*math.cos(case['angle']);y=radius*math.sin(case['angle'])
 cmd=[str(R/'work/g1_sim_env/bin/python'),str(R/'work/table_run_demo_20261005.py'),'--name',case['name'],'--seed',str(case['seed']),'--table-x',str(x),'--table-y',str(y)]
 with (out/(case['name']+'.log')).open('w') as log:subprocess.run(cmd,cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
 result=json.loads((D/case['name']/'scene_000/unified/result.json').read_text());assert result['checkpoint_sha256']==f['tracker_sha256'] and result['scene']['weight_sha256']==f['goal_sha256'] and result['scene']['seed']==case['seed'];results.append(dict(case=case,result=result));(out/'results.json').write_text(json.dumps(results,indent=2));print(case['name'],result['success'],result['palm_target_error_m'],flush=True)
(out/'complete.json').write_text(json.dumps(dict(planned=6,successes=sum(x['result']['success'] for x in results),same_generation_noise_within_each_group=True,same_frozen_tracker_for_all=True),indent=2))
