from pathlib import Path
import subprocess,os
R=Path('/home/pku/frankenmotion');W=R/'work/universal_tracker_20261006';D=R/'outputs_amass/universal_tracker_20261006'
commands=[[str(R/'work/g1_sim_env/bin/python'),str(W/'retarget_natural_candidates.py')],[str(R/'work/mjlab_stable_env/bin/python'),str(W/'prepare_natural_motion.py')],[str(R/'work/mjlab_stable_env/bin/python'),str(W/'evaluate_cpu_general.py'),'--checkpoint',str(D/'backup/prior_unified_frozen/policy.pt'),'--manifest',str(D/'generated_demo_motion/manifest.json'),'--name','generated_baseline_broad']]
for command in commands:subprocess.run(command,env=dict(os.environ,NATURAL_SET='generated_demo_candidates',NATURAL_MOTION_SET='generated_demo_motion'),check=True,cwd=R)
