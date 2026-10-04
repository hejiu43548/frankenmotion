"""Technical correction only: fixed models/inputs, enlarge collision allocation."""
from pathlib import Path
import os,json,subprocess,datetime,hashlib
R=Path('/home/pku/frankenmotion');N=R/'outputs_amass/franken_improve_20261003';W=R/'work';script=W/'mjlab_probe_capacity_20261003.py'
protocol=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),reason='MuJoCo-Warp broadphase overflow in V1 BM and novel-text logs; matching cached metrics does not rule out physical contact loss',change='nconmax=256,njmax=2048 evaluation allocation only',unchanged='Frozen weights, routes, commands, generated motions, robot gains, timestep and task metrics',script_sha256=hashlib.sha256(script.read_bytes()).hexdigest(),published_v1='Preserve historical archive and provide correction notice; do not silently replace',scopes=['V1 BM880','integrated V3 BM240','novel phrasing BM30'])
(N/'capacity_correction_protocol.json').write_text(json.dumps(protocol,indent=2))
cases=[('integrated_v3',N/'integrated_v3_manifest.json',N/'integrated_v3_sonic','reach,sidestep,back_walk','integrated_v3_beyondmimic_capacity256'),('novel_v3',N/'novel_prompt_v3/manifest.json',N/'novel_prompt_v3_sonic','reach,sidestep,back_walk','novel_prompt_v3_beyondmimic_capacity256'),('v1',N/'candidate_confirmation_manifest.json',N/'confirmation/candidate','','beyondmimic_confirmation_capacity256')]
for label,manifest,refs,tasks,out in cases:
 env=dict(os.environ,BM_CPU_FK='1',BM_ENTRY='standing',BM_TERMINATION='physical',BM_TASK=tasks,BM_RESULTS=str(manifest),BM_REFERENCE_DIR=str(refs),BM_WEIGHT_MAP=str(N/'frozen_controllers_v1/weight_map.json'),BM_QUIET_METRICS='1',BM_OUT=out)
 print('START',label,flush=True)
 with (W/('capacity_'+label+'.log')).open('w') as f:subprocess.run([str(W/'mjlab_stable_env/bin/python'),str(script)],env=env,cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
 text=(W/('capacity_'+label+'.log')).read_text();assert 'overflow' not in text.lower(),label
 print('DONE',label,flush=True)
