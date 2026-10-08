"""Matched-entry development diagnostic; not used for selection or final scoring."""
import json,time,hashlib,subprocess
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/universal_tracker_20261006';W=R/'work/universal_tracker_20261006';py=R/'work/mjlab_stable_env/bin/python'
for label,step in [('v5_1000',1000),('v5_3000',3000),('v5_final',5999)]:
 ck=D/f'training/joint_v5_expanded/model_{step}.pt'
 while not ck.with_suffix('.pt.ready.json').exists():
  if time.time()>1791247800:raise RuntimeError('Diagnostic deadline')
  time.sleep(10)
 assert hashlib.sha256(ck.read_bytes()).hexdigest()==json.loads(ck.with_suffix('.pt.ready.json').read_text())['checkpoint_sha256']
 name='validation_rsi_'+label
 with (D/(name+'.log')).open('w') as log:
  subprocess.run([str(py),str(W/'evaluate_cpu_general.py'),'--checkpoint',str(ck),'--manifest',str(D/'natural_extended_corrected_motion/manifest.json'),'--split','val','--entry','rsi','--name',name,'--workers','4'],cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
  subprocess.run([str(py),str(W/'assess_general_fidelity.py'),'--run',str(D/'general_evaluation'/name)],cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
 print(name,'complete',flush=True)
