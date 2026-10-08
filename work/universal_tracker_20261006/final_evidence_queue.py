"""Assemble evidence after declared tests finish; visual review remains manual."""
import json,time,subprocess,datetime
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/universal_tracker_20261006';W=R/'work/universal_tracker_20261006';py=R/'work/mjlab_stable_env/bin/python'
cutoff=datetime.datetime(2026,10,6,2,10,tzinfo=datetime.timezone.utc).timestamp()
while True:
 p=D/'frozen_test_supervisor_status.json';status=json.loads(p.read_text()) if p.exists() else {}
 if any(v['returncode'] for v in status.values()):raise RuntimeError(('Evaluation phase failed; inspect before evidence assembly',status))
 if set(status)=={'cpu','gpu','robustness','demos'}:break
 if time.time()>cutoff:raise RuntimeError('Assembly timeout; partial evidence must be reported explicitly')
 time.sleep(10)
for script in ['verify_single_frozen_policy.py','collect_release_inventory.py','collect_report_data.py']:
 subprocess.run([str(py),str(W/script)],cwd=R,check=True)
print('Evidence assembled; publication figures, video inspection and PDF review still required.',flush=True)
