"""Time-bounded study orchestration; development selection never reads final results."""
import time,json,datetime,subprocess
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/universal_tracker_20261006';W=R/'work/universal_tracker_20261006';py=R/'work/mjlab_stable_env/bin/python'
def now():return datetime.datetime.now(datetime.timezone.utc).isoformat()
deadline=datetime.datetime(2026,10,6,0,35,tzinfo=datetime.timezone.utc).timestamp()
while time.time()<deadline and not (D/'frozen_unified/protocol.json').exists():time.sleep(min(10,deadline-time.time()))
if not (D/'frozen_unified/protocol.json').exists():
 with (D/'freeze_candidate.log').open('w') as log:subprocess.run([str(py),str(W/'freeze_candidate.py')],cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
print('Policy frozen',now(),flush=True)
processes={};status={}
def launch(phase):
 command=[str(py),str(W/('evaluate_frozen_demo_candidates.py' if phase=='demos' else 'run_frozen_tests.py'))]+([] if phase=='demos' else ['--phase',phase])
 with (D/('frozen_'+phase+'.log')).open('w') as log:return subprocess.Popen(command,cwd=R,stdout=log,stderr=subprocess.STDOUT)
def poll():
 for phase,process in list(processes.items()):
  result=process.poll()
  if result is not None:
   status[phase]=dict(returncode=result,finished_utc=now());del processes[phase];(D/'frozen_test_supervisor_status.json').write_text(json.dumps(status,indent=2));print(phase,status[phase],flush=True)
# Existing generated demo inputs are independent of fresh-test preparation.
processes['demos']=launch('demos');prepare=launch('prepare')
while prepare.poll() is None:poll();time.sleep(10)
(D/'frozen_preparation_status.json').write_text(json.dumps(dict(returncode=prepare.returncode,finished_utc=now()),indent=2))
if prepare.returncode:raise RuntimeError(('Fresh preparation failed',prepare.returncode))
print('Fresh references prepared',now(),flush=True)
for phase in ['cpu','gpu','robustness']:processes[phase]=launch(phase)
while processes:poll();time.sleep(10) if processes else None
if any(v['returncode'] for v in status.values()):raise RuntimeError(status)
print('All declared evaluation stages finished. Human-facing analysis and visual checks still required.',flush=True)
