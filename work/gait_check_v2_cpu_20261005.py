import subprocess,json
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/gait_demo_20261005';py=str(R/'work/mjlab_stable_env/bin/python')
for i in range(2,6):
 s=D/f'dev_timed/scene_{i:03d}';out=s/'distill_v2_3000_cpu'
 subprocess.run([py,str(R/'work/gait_cpu_evaluate_20261005.py'),'--run',str(s/'baseline_timed'),'--actor',str(D/'training/distill_v2/actor_3000.pt'),'--checkpoint',str(D/'training/distill_v2/model_3000.pt'),'--output',str(out)],check=True)
 subprocess.run([py,str(R/'work/gait_metrics_20261005.py'),'--run',str(out)],check=True)
