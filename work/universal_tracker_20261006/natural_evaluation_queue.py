"""CPU development checks as checkpoints become immutable; not final testing."""
import json,time,subprocess,hashlib,argparse
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/universal_tracker_20261006';W=R/'work/universal_tracker_20261006';py=R/'work/mjlab_stable_env/bin/python'
p=argparse.ArgumentParser();p.add_argument('--spec',required=True);a=p.parse_args();items=json.loads(Path(a.spec).read_text())
for item in items:
 ck=Path(item['checkpoint']);ready=ck.with_suffix('.pt.ready.json')
 while not ready.exists():
  if time.time()>1791247800:raise RuntimeError('No new development runs after00:50UTC')
  time.sleep(10)
 assert hashlib.sha256(ck.read_bytes()).hexdigest()==json.loads(ready.read_text())['checkpoint_sha256']
 name=item['name'];commands=[]
 for tag,dataset in [('natural','natural_extended_corrected_motion'),('locomotion','natural_locomotion_motion')]:
  output=f'{tag}_{name}';path=D/'general_evaluation'/output
  if not path.exists():commands.append([str(py),str(W/'evaluate_cpu_general.py'),'--checkpoint',str(ck),'--manifest',str(D/dataset/'manifest.json'),'--split','val','--name',output,'--workers','4'])
  if not (path/'fidelity_summary.json').exists():commands.append([str(py),str(W/'assess_general_fidelity.py'),'--run',str(path)])
 if not (D/'table_evaluation'/name/'summary.json').exists():commands.append([str(py),str(W/'evaluate_table.py'),'--checkpoint',str(ck),'--name',name,'--workers','2'])
 for i,c in enumerate(commands):
  with (D/f'{name}_natural_queue_{i}.log').open('w') as log:result=subprocess.run(c,cwd=R,stdout=log,stderr=subprocess.STDOUT)
  if result.returncode:raise RuntimeError((name,i,result.returncode))
 print(name,'CPU natural/table checks complete',flush=True)
