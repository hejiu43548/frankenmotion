from pathlib import Path
import subprocess,os
R=Path('/home/pku/frankenmotion');W=R/'work/universal_tracker_20261006';p=str(R/'work/mjlab_stable_env/bin/python');g=str(R/'work/g1_sim_env/bin/python')
commands=[[g,str(W/'prepare_locomotion_inventory.py')],[p,str(W/'prepare_natural_candidates.py'),'--name','natural_locomotion','--inventory','locomotion_inventory.json','--train-limit','30','--val-limit','5','--max-frames','300'],[g,str(W/'retarget_natural_candidates.py')],[p,str(W/'prepare_natural_motion.py')]]
for command in commands:subprocess.run(command,env=dict(os.environ,NATURAL_SET='natural_locomotion',NATURAL_MOTION_SET='natural_locomotion_motion'),check=True,cwd=R)
