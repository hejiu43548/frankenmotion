from pathlib import Path
import json,subprocess
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/universal_tracker_20261006';W=R/'work/universal_tracker_20261006';py=str(R/'work/mjlab_stable_env/bin/python')
for old in sorted((D/'general_evaluation').iterdir()):
 if not (old/'INVALIDATED.json').exists():continue
 p=json.loads((old/'protocol.json').read_text());name=old.name+'_flat_v2';cmd=[py,str(W/'evaluate_cpu_general.py'),'--checkpoint',p['checkpoint'],'--manifest',p['manifest'],'--name',name,'--entry','rsi' if p.get('entry')=='rsi' else 'standing']
 if p.get('split'):cmd.extend(['--split',p['split']])
 with (D/(name+'.log')).open('w') as log:subprocess.run(cmd,cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
 if name.startswith('eleven_'):
  with (D/(name+'_assess.log')).open('w') as log:subprocess.run([py,str(W/'assess_native_eleven.py'),'--name',name],cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
 print(name,'complete',flush=True)
