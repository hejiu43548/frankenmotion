"""Time-bounded study orchestration; development selection never reads final results."""
import sys,time,json,datetime,subprocess
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/universal_tracker_20261006';W=R/'work/universal_tracker_20261006';py=R/'work/mjlab_stable_env/bin/python'
deadline=datetime.datetime(2026,10,6,0,35,tzinfo=datetime.timezone.utc).timestamp()
while time.time()<deadline and not (D/'frozen_unified/protocol.json').exists():time.sleep(min(10,deadline-time.time()))
if not (D/'frozen_unified/protocol.json').exists():
 with (D/'freeze_candidate.log').open('w') as log:subprocess.run([str(py),str(W/'freeze_candidate.py')],cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
print('Policy frozen',datetime.datetime.now(datetime.timezone.utc).isoformat(),flush=True)
with (D/'frozen_prepare.log').open('w') as log:subprocess.run([str(py),str(W/'run_frozen_tests.py'),'--phase','prepare'],cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
print('Fresh references prepared',datetime.datetime.now(datetime.timezone.utc).isoformat(),flush=True)
processes={}
for phase in ['cpu','gpu','robustness','demos']:
 command=[str(py),str(W/('evaluate_frozen_demo_candidates.py' if phase=='demos' else 'run_frozen_tests.py'))]+([] if phase=='demos' else ['--phase',phase])
 log=(D/('frozen_'+phase+'.log')).open('w');processes[phase]=subprocess.Popen(command,cwd=R,stdout=log,stderr=subprocess.STDOUT);log.close()
status={}
while processes:
 for phase,process in list(processes.items()):
  result=process.poll()
  if result is not None:
   status[phase]=dict(returncode=result,finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat());del processes[phase];(D/'frozen_test_supervisor_status.json').write_text(json.dumps(status,indent=2));print(phase,status[phase],flush=True)
 if processes:time.sleep(10)
if any(v['returncode'] for v in status.values()):raise RuntimeError(status)
print('All declared evaluation stages finished. Human-facing analysis and visual checks still required.',flush=True)
