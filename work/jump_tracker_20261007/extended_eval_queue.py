import sys,time,json,subprocess
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/jump_tracker_20261007';W=R/'work/jump_tracker_20261007';py=sys.executable
for name,steps in [('ballistic_shared_3000',[1000,2000,2999]),('envelope_shared_4000',[1000,2000,3000,3999])]:
 for step in steps:
  checkpoint=D/'training'/name/f'model_{step}.pt'
  while not checkpoint.with_suffix('.pt.ready.json').exists():
   status=D/(name+'_process.json')
   if status.exists() and json.loads(status.read_text()).get('returncode',0)!=0:raise RuntimeError(name+' failed')
   time.sleep(5)
  for dataset in (['jump','ballistic'] if 'ballistic' in name else ['jump']):
   tag=f'{name}_{step}_{dataset}';subprocess.run([py,str(W/'evaluate.py'),'--checkpoint',str(checkpoint),'--manifest',str(D/f'dev_{dataset}_manifest.json'),'--name',tag,'--workers','4'],check=True)
   subprocess.run([py,str(W/('assess_timing.py' if dataset=='ballistic' else 'assess.py')),'--name',tag],check=True)
