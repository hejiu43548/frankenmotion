"""Orchestrate fixed integrated benchmark; no outcome-based selection."""
from pathlib import Path
import subprocess,json,os
ROOT=Path('/home/pku/frankenmotion');NEW=ROOT/'outputs_amass/franken_improve_20261003';W=ROOT/'work';GEN=ROOT/'.conda/bin/python';SIM=W/'g1_sim_env/bin/python';BM=W/'mjlab_stable_env/bin/python'
def run(py,script,args=[],env=None):
 print('START',script,args,flush=True)
 with (W/(script+'_log.txt')).open('w') as f:subprocess.run([str(py),str(W/(script+'_20261003.py')),*args],cwd=ROOT,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('DONE',script,flush=True)
physical='raise_hand,reach,strike,wave,turn,sidestep,back_walk,lean,walk'
for label,tasks,weight in [('integrated_v3_physical',physical,'frozen_generation_v1/physical.pt'),('integrated_v3_jump','jump','frozen_generation_v1/jump_aligned.pt'),('integrated_v3_kick','kick','frozen_kick_v3/kick.pt'),('integrated_v3_control_kick','kick','frozen_generation_v1/physical.pt')]:
 run(GEN,'generate_v3',['--phase','confirmation','--weights',str(NEW/weight),'--label',label,'--tasks',tasks])
rows=[]
for label in ['integrated_v3_physical','integrated_v3_jump','integrated_v3_kick']:rows+=json.loads((NEW/(label+'_confirmation_manifest.json')).read_text())
assert len(rows)==880;(NEW/'integrated_v3_manifest.json').write_text(json.dumps(rows,indent=2));control=[r for r in rows if r['task'] in ['raise_hand','lean']]+json.loads((NEW/'integrated_v3_control_kick_confirmation_manifest.json').read_text());assert len(control)==240;(NEW/'integrated_v3_control_manifest.json').write_text(json.dumps(control,indent=2))
run(SIM,'eval_integrated_v3')
run(SIM,'confirmation_eval',['--manifest',str(NEW/'integrated_v3_control_manifest.json'),'--name','integrated_v3_control','--methods','uniform','--workers','4'])
env=dict(os.environ,BM_CPU_FK='1',BM_ENTRY='standing',BM_TERMINATION='physical',BM_TASK='reach,sidestep,back_walk',BM_RESULTS=str(NEW/'integrated_v3_manifest.json'),BM_REFERENCE_DIR=str(NEW/'integrated_v3_sonic'),BM_WEIGHT_MAP=str(NEW/'frozen_controllers_v1/weight_map.json'),BM_QUIET_METRICS='1',BM_OUT='integrated_v3_beyondmimic')
run(BM,'mjlab_probe',env=env)
print('INTEGRATED V3 PHYSICAL COMPLETE: 1120 unique rollouts, assess manually',flush=True)
