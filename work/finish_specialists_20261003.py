"""Evaluate the predeclared final training steps on development only."""
from pathlib import Path
import os,time,json,subprocess
R=Path('/home/pku/frankenmotion');N=R/'outputs_amass/franken_improve_20261003';W=R/'work'
for pid in [354028,354096]:
 while True:
  try:os.kill(pid,0)
  except ProcessLookupError:break
  time.sleep(5)
for task,suffix in [('turn','ori2.0_vel1.0'),('strike','ori0.5_vel2.0')]:
 folder=N/('beyondmimic_finetune_'+task+'_all_'+suffix);assert json.loads((folder/'complete.json').read_text())['iterations']==2400;assert (folder/'model_2399.pt').exists()
cases=[('turn',N/'beyondmimic_finetune_turn_all_ori2.0_vel1.0/model_2399.pt','specialist_turn_development_capacity256'),('strike',N/'beyondmimic_finetune_strike_all_ori0.5_vel2.0/model_2399.pt','specialist_strike_development_capacity256'),('turn,strike',W/'beyondmimic_demo.pt','specialist_dance_control_capacity256')]
for task,ckpt,out in cases:
 env=dict(os.environ,BM_CPU_FK='1',BM_ENTRY='standing',BM_TERMINATION='physical',BM_TASK=task,BM_OUT=out,BM_CHECKPOINT=str(ckpt),BM_QUIET_METRICS='1');log=W/(out+'.log');print('START',out,flush=True)
 with log.open('w') as f:subprocess.run([str(W/'mjlab_stable_env/bin/python'),str(W/'mjlab_probe_capacity_20261003.py')],cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 assert 'overflow' not in log.read_text().lower();print('DONE',out,flush=True)
print('Development specialist comparisons complete; no independent claim',flush=True)
