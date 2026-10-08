"""Launch an authorized experiment after an earlier GPU worker exits."""
import argparse,json,time,subprocess
from pathlib import Path
p=argparse.ArgumentParser();p.add_argument('--spec',required=True);a=p.parse_args();spec=json.loads(Path(a.spec).read_text());D=Path('/home/pku/frankenmotion/outputs_amass/universal_tracker_20261006');R=Path('/home/pku/frankenmotion')
dependencies=spec.get('dependencies',[spec['dependency']])
while not all(Path(x).exists() for x in dependencies):
 if time.time()>1791246000:raise RuntimeError('Too late to start another training experiment in this window')
 time.sleep(10)
for key in ['smoke','train']:
 command=spec[key]
 with (D/(spec['name']+'_'+key+'.log')).open('w') as log:
  result=subprocess.run(command,cwd=R,stdout=log,stderr=subprocess.STDOUT)
 (D/(spec['name']+'_'+key+'_status.json')).write_text(json.dumps(dict(command=command,returncode=result.returncode),indent=2))
 if result.returncode:raise RuntimeError(key+' failed; do not start dependent run')
 print(key,'complete',flush=True)
