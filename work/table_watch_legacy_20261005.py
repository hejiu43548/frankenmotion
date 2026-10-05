import os,sys,json,time,subprocess,argparse
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/table_demo_20261005';U=R/'outputs_amass/franken_unified_20261004'
p=argparse.ArgumentParser();p.add_argument('--pid',required=True,type=int);a=p.parse_args();folder=D/'training/retention_v2';results=[]
for step in [500,1000,1499]:
 checkpoint=folder/f'model_{step}.pt'
 while not checkpoint.with_suffix('.pt.ready.json').exists():
  if not Path(f'/proc/{a.pid}').exists():raise RuntimeError('Training exited without checkpoint '+str(step))
  time.sleep(10)
 name=f'table_retention_v2_{step}_20261005'
 subprocess.run([str(R/'work/mjlab_stable_env/bin/python'),str(R/'work/unified_evaluate_20261004.py'),'--checkpoint',str(checkpoint),'--preview','--name',name],cwd=R,check=True)
 subprocess.run([str(R/'work/g1_sim_env/bin/python'),str(R/'work/unified_assess_20261004.py'),'--name',name],cwd=R,check=True)
 audit=json.loads((U/'evaluation'/name/'audit.json').read_text());results.append(dict(step=step,name=name,audit=audit));(folder/'legacy_results.json').write_text(json.dumps(results,indent=2));print(step,audit['aggregate'],flush=True)
(folder/'legacy_complete.json').write_text(json.dumps(dict(candidates=len(results),requests_each=110)))
