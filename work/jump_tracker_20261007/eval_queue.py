import sys,time,json,subprocess
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/jump_tracker_20261007';W=R/'work/jump_tracker_20261007';py=sys.executable
for name in ['phase_shared_3000','exposure_control_3000']:
 for step in [1000,2000,2999]:
  checkpoint=D/'training'/name/f'model_{step}.pt'
  while not checkpoint.with_suffix('.pt.ready.json').exists():
   status=D/(name+'_process.json')
   if status.exists() and json.loads(status.read_text()).get('returncode',0)!=0:raise RuntimeError(name+' failed')
   time.sleep(5)
  tag=f'{name}_{step}'
  subprocess.run([py,str(W/'evaluate.py'),'--checkpoint',str(checkpoint),'--manifest',str(D/'dev_jump_manifest.json'),'--name',tag,'--workers','4'],check=True)
  subprocess.run([py,str(W/'assess.py'),'--name',tag],check=True)
