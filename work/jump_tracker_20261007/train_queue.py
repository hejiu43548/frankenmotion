import sys,time,json,subprocess
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/jump_tracker_20261007';W=R/'work/jump_tracker_20261007'
s=D/'phase_shared_3000_process.json'
while not s.exists() or 'returncode' not in json.loads(s.read_text()):time.sleep(5)
assert json.loads(s.read_text())['returncode']==0
subprocess.run([sys.executable,str(W/'launch_training.py'),'exposure_control_3000','0','3000'],check=True)
