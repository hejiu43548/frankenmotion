import sys,json,time,subprocess,shutil
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/unified_generator_20261007';W=R/'work/unified_generator_20261007';py=R/'work/mjlab_stable_env/bin/python';actor=D/'backup/tracker_candidate_actor.pt';shutil.copy2(R/'outputs_amass/jump_tracker_20261007/final_candidate/actor.json',actor.with_suffix('.json'))
for name in ['shared_v3_4400_development','shared_v3_22000_development','shared_v4_4400_development','shared_v4_11000_development']:

 while not (D/'generation'/name/'complete.json').exists():time.sleep(5)
 manifest=D/'generation'/name/'native_manifest.json'
 if not manifest.exists():
  with (D/(name+'_retarget.log')).open('w') as f:subprocess.run([str(py),str(W/'prepare_tracking.py'),'--name',name],stdout=f,stderr=subprocess.STDOUT,check=True)
 with (D/(name+'_tracking.log')).open('w') as f:subprocess.run([str(py),str(W/'evaluate_tracker.py'),'--checkpoint',str(actor),'--actor',str(actor),'--manifest',str(manifest),'--name',name,'--workers','4'],stdout=f,stderr=subprocess.STDOUT,check=True)
 subprocess.run([str(py),str(W/'assess_tracker.py'),'--name',name],check=True)
