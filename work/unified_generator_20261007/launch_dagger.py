import sys,time,subprocess,json
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/unified_generator_20261007';W=R/'work/unified_generator_20261007';initial=D/'training/shared_v3_wide/model_22000.pt'
while not initial.with_suffix('.ready.json').exists():time.sleep(5)
with (D/'shared_v4_dagger.log').open('w') as f:
 p=subprocess.run([sys.executable,str(W/'train_dagger.py'),'--name','shared_v4_dagger','--steps','11000','--lr','0.00001','--initial',str(initial),'--recon-weight','0','--command-weight','0'],stdout=f,stderr=subprocess.STDOUT,cwd=R)
(D/'shared_v4_dagger_process.json').write_text(json.dumps(dict(returncode=p.returncode)))
