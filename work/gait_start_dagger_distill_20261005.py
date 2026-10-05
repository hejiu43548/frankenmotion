import time,json,subprocess
from pathlib import Path
import numpy as np
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/gait_demo_20261005'
while not (D/'teacher_data_dagger1/protocol.json').exists():time.sleep(5)
a=np.load(D/'teacher_data/dataset.npz');b=np.load(D/'teacher_data_dagger1/dataset.npz');out=D/'teacher_data_combined';out.mkdir(exist_ok=False);np.savez_compressed(out/'dataset.npz',**{k:np.concatenate([a[k],b[k]]) for k in a.files});(out/'protocol.json').write_text(json.dumps(dict(original_groups=16,new_dagger_groups=list(range(8)),validation_groups=[14,15],validation_refs_not_in_dagger=True)))
with (R/'work/gait_distill_v2.log').open('w') as f:subprocess.run([str(R/'work/mjlab_stable_env/bin/python'),str(R/'work/gait_distill_20261005.py'),'--name','distill_v2','--dataset','teacher_data_combined/dataset.npz','--initial',str(D/'training/distill_v1/model_3000.pt')],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
subprocess.run([str(R/'work/mjlab_stable_env/bin/python'),str(R/'work/unified_export_actor_20261004.py'),'--checkpoint',str(D/'training/distill_v2/model_3000.pt'),'--output',str(D/'training/distill_v2/actor_3000.pt')],cwd=R,check=True)
for i in [0,1]:
 src=D/f'dev_timed/scene_{i:03d}/baseline_timed';dest=src.parent/'distill_v2_3000_cpu'
 subprocess.run([str(R/'work/mjlab_stable_env/bin/python'),str(R/'work/gait_cpu_evaluate_20261005.py'),'--run',str(src),'--actor',str(D/'training/distill_v2/actor_3000.pt'),'--checkpoint',str(D/'training/distill_v2/model_3000.pt'),'--output',str(dest)],cwd=R,check=True)
 subprocess.run([str(R/'work/mjlab_stable_env/bin/python'),str(R/'work/gait_metrics_20261005.py'),'--run',str(dest)],cwd=R,check=True)
