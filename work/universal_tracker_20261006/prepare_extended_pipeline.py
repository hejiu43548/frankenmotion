from pathlib import Path
import subprocess,os
R=Path('/home/pku/frankenmotion');W=R/'work/universal_tracker_20261006'
commands=[[str(R/'work/mjlab_stable_env/bin/python'),str(W/'prepare_natural_candidates.py'),'--name','natural_extended','--train-limit','20','--val-limit','5','--max-frames','300'],[str(R/'work/g1_sim_env/bin/python'),str(W/'retarget_natural_candidates.py')],[str(R/'work/mjlab_stable_env/bin/python'),str(W/'prepare_natural_motion.py')]]
for command in commands:subprocess.run(command,env=dict(os.environ,NATURAL_SET='natural_extended',NATURAL_MOTION_SET='natural_extended_motion'),check=True,cwd=R)
