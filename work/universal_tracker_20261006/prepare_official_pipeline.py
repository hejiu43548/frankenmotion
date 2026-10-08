from pathlib import Path
import subprocess,os
R=Path('/home/pku/frankenmotion');W=R/'work/universal_tracker_20261006';D=R/'outputs_amass/universal_tracker_20261006';py=str(R/'work/mjlab_stable_env/bin/python')
commands=[[str(R/'work/g1_sim_env/bin/python'),str(W/'retarget_natural_candidates.py')],[py,str(W/'prepare_natural_motion.py')],[py,str(W/'evaluate_cpu_general.py'),'--checkpoint',str(D/'backup/prior_unified_frozen/policy.pt'),'--manifest',str(D/'generated_official_motion/manifest.json'),'--name','official_text_broad_flat_v2']]
for command in commands:subprocess.run(command,env=dict(os.environ,NATURAL_SET='generated_official_text',NATURAL_MOTION_SET='generated_official_motion'),check=True,cwd=R)
