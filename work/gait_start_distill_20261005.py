import time,subprocess
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/gait_demo_20261005'
while not (D/'teacher_data/protocol.json').exists():time.sleep(10)
with (R/'work/gait_distill_v1.log').open('w') as f:subprocess.run([str(R/'work/mjlab_stable_env/bin/python'),str(R/'work/gait_distill_20261005.py')],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
