import time,subprocess
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/turn_demo_20261005';py=str(R/'work/mjlab_stable_env/bin/python');start=time.monotonic()
while not (D/'full_teacher_v1/protocol.json').exists():
 if time.monotonic()-start>1800:raise RuntimeError('Teacher dataset was not completed within 30 minutes')
 time.sleep(10)
subprocess.run([py,str(R/'work/turn_distill_20261005.py'),'--name','distill_full_v1','--datasets','teacher_v1','full_teacher_v1','--initial',str(D/'distill_v1/model_4000.pt'),'--steps','6000'],check=True)
subprocess.run([py,str(R/'work/unified_export_actor_20261004.py'),'--checkpoint',str(D/'distill_full_v1/model_6000.pt'),'--output',str(D/'distill_full_v1/actor_6000.pt')],check=True)
subprocess.run([py,str(R/'work/turn_evaluate_20261005.py'),'--folder',str(D/'development_v2'),'--checkpoint',str(D/'distill_full_v1/model_6000.pt'),'--name','distill_full_v1'],check=True)
