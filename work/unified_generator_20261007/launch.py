import sys,time,json,subprocess
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/unified_generator_20261007';W=R/'work/unified_generator_20261007'
while not (D/'teacher_training/complete.json').exists():time.sleep(5)
with (D/'shared_v1.log').open('w') as f:
 p=subprocess.run([sys.executable,str(W/'train_joint.py'),'--name','shared_v1','--steps','11000'],cwd=R,stdout=f,stderr=subprocess.STDOUT)
(D/'shared_v1_process.json').write_text(json.dumps(dict(returncode=p.returncode)))
