import time,subprocess
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/gait_demo_20261005'
while not (D/'training/feet_v1/complete.json').exists():time.sleep(15)
cmd=[str(R/'work/mjlab_stable_env/bin/python'),str(R/'work/gait_train_v2_20261005.py'),'--name','sole_v2','--dataset','training_corpus_timed','--preview','--steps','2500','--envs','448','--seed','71005002','--initial',str(D/'training/feet_v1/model_1000.pt'),'--root-weight','2','--root-std','.15','--body-pos-weight','2','--body-pos-std','.1','--joint-weight','2','--joint-std','.3','--joint-worst-count','8','--action-rate-weight','-.01','--learning-rate','.00005','--entropy-coef','.003','--episode-seconds','25','--root-ori-weight','2','--root-ori-std','.4','--root-wide-weight','1','--root-wide-std','.5','--task-balanced-slots','--reset-training-rng','--table-fraction','.5','--hand-weight','.5']
with (R/'work/gait_train_sole_v2.log').open('w') as f:subprocess.run(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
print('SOLE V2 COMPLETE')
