"""Run predeclared development checks as immutable checkpoints become ready."""
import json,time,subprocess,datetime,argparse,hashlib
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/universal_tracker_20261006';W=R/'work/universal_tracker_20261006'
p=argparse.ArgumentParser();p.add_argument('--spec',required=True);a=p.parse_args();spec=json.loads(Path(a.spec).read_text())
py=str(R/'work/mjlab_stable_env/bin/python');metricpy=str(R/'work/g1_sim_env/bin/python')
for item in spec:
 ck=Path(item['checkpoint']);name=item['name'];ready=ck.with_suffix('.pt.ready.json')
 while not ready.exists():
  if time.time()>1791253153:raise RuntimeError('Stop new evaluation near end of authorized window')
  time.sleep(10)
 assert hashlib.sha256(ck.read_bytes()).hexdigest()==json.loads(ready.read_text())['checkpoint_sha256']
 print(datetime.datetime.now().isoformat(),name,'starting',flush=True)
 commands=[('evaluate',[py,str(W/'evaluate.py'),'--checkpoint',str(ck),'--preview','--name',name]),('assess',[metricpy,str(W/'assess.py'),'--name',name]),('table',[py,str(W/'evaluate_table.py'),'--checkpoint',str(ck),'--name',name])]
 status={}
 for phase,command in commands:
  existing=D/'table_evaluation'/name/'summary.json'
  if phase=='table' and existing.exists():
   cached=json.loads(existing.read_text());assert cached['checkpoint_sha256']==hashlib.sha256(ck.read_bytes()).hexdigest() and cached['processed']==cached['planned'];status[phase]=0;continue
  with (D/(name+'_'+phase+'.log')).open('w') as log:r=subprocess.run(command,cwd=R,stdout=log,stderr=subprocess.STDOUT)
  status[phase]=r.returncode
  if r.returncode:print(name,phase,'FAILED',r.returncode,flush=True);break
 (D/(name+'_queue_status.json')).write_text(json.dumps(status,indent=2))
 print(datetime.datetime.now().isoformat(),name,status,flush=True)
