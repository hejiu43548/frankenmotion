"""Wait for known live development jobs, freeze selection, then run confirmation.
No confirmation metric is consulted for controller selection.
"""
import os,time,json,subprocess
from pathlib import Path
ROOT=Path('/home/pku/frankenmotion/outputs_amass/franken_improve_20261003');WORK=Path('/home/pku/frankenmotion/work');PY=WORK/'mjlab_stable_env/bin/python';SIM=WORK/'g1_sim_env/bin/python';PLOT='/home/pku/frankenmotion/outputs_amass/g1_command_diagnosis_20261002_2346/plot_deps'
def run(script,env=None,log=None,python=PY):
 cmd=[str(python),str(WORK/(script+'_20261003.py'))];print('START',script,flush=True)
 if log:
  with (WORK/log).open('w') as stream:subprocess.run(cmd,env=env,stdout=stream,stderr=subprocess.STDOUT,check=True)
 else:subprocess.run(cmd,env=env,check=True)
 print('DONE',script,flush=True)
for suffix,pid,trained,out in [('std03',301710,'beyondmimic_finetune_sidestep_root2.0_all','beyondmimic_side_std03_development'),('std01',302776,'beyondmimic_finetune_sidestep_root2.0_all_std0.1','beyondmimic_side_std01_development')]:
 deadline=time.monotonic()+7200
 while Path('/proc/'+str(pid)).exists():
  if time.monotonic()>deadline:raise TimeoutError('Development process still active after two hours')
  time.sleep(10)
 complete=json.loads((ROOT/trained/'complete.json').read_text());assert complete['iterations']==2400
 env=os.environ.copy();env.update(BM_CPU_FK='1',BM_ENTRY='standing',BM_TERMINATION='physical',BM_TASK='sidestep',BM_OUT=out,BM_CHECKPOINT=str(ROOT/trained/'model_2399.pt'),OPENBLAS_NUM_THREADS='1')
 run('mjlab_probe',env,'mjlab_side_'+suffix+'_development_20261003.log')
run('freeze_controllers',python=SIM)
assert (ROOT/'confirmation/candidate/results.json').exists() and (ROOT/'confirmation/paired_baseline/results.json').exists()
env=os.environ.copy();env.update(BM_CPU_FK='1',BM_ENTRY='standing',BM_TERMINATION='physical',BM_OUT='beyondmimic_confirmation',BM_RESULTS=str(ROOT/'candidate_confirmation_manifest.json'),BM_REFERENCE_DIR=str(ROOT/'confirmation/candidate'),BM_WEIGHT_MAP=str(ROOT/'frozen_controllers_v1/weight_map.json'),BM_CHECKPOINT=str(ROOT/'frozen_controllers_v1/dance.pt'),BM_QUIET_METRICS='1',OPENBLAS_NUM_THREADS='1');env.pop('BM_TASK',None)
run('mjlab_probe',env,'beyondmimic_confirmation_20261003.log')
run('recheck_trajectories',python=SIM)
run('assess_confirmation',python=SIM)
env=os.environ.copy();env['PYTHONPATH']=PLOT;run('plot_confirmation',env,python=SIM)
print('CONFIRMATION DATA AND FIGURES COMPLETE; manual review/report/package still required',flush=True)
