from pathlib import Path
import subprocess,os
R=Path('/home/pku/frankenmotion');W=R/'work/universal_tracker_20261006';D=R/'outputs_amass/universal_tracker_20261006';py=str(R/'work/mjlab_stable_env/bin/python')
commands=[[str(R/'work/g1_sim_env/bin/python'),str(W/'retarget_natural_candidates.py')],[py,str(W/'prepare_natural_motion.py')]]
for name,ck in [('refined_baseline_broad',D/'backup/prior_unified_frozen/policy.pt'),('refined_v1_final',D/'training/joint_v1_stable_init/model_1499.pt'),('refined_v2_1000',D/'training/joint_v2_shared_global/model_1000.pt')]:commands.append([py,str(W/'evaluate_cpu_general.py'),'--checkpoint',str(ck),'--manifest',str(D/'generated_refined_motion/manifest.json'),'--name',name])
for command in commands:subprocess.run(command,env=dict(os.environ,NATURAL_SET='generated_composed_refined',NATURAL_MOTION_SET='generated_refined_motion'),check=True,cwd=R)
