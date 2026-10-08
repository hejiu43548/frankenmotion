import sys,time,json,subprocess
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/jump_tracker_20261007';W=R/'work/jump_tracker_20261007';py=sys.executable
p=D/'exposure_control_3000_process.json'
while not p.exists() or 'returncode' not in json.loads(p.read_text()):time.sleep(5)
assert json.loads(p.read_text())['returncode']==0
for name,phase,override in [('ballistic_shared_3000',1,'ballistic_training_override.json'),('envelope_shared_4000',1,'envelope_training_override.json')]:
 subprocess.run([py,str(W/'launch_training.py'),name,str(phase),'4000' if 'envelope' in name else '3000',str(D/override)],check=True)
