import sys,json,subprocess
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/jump_tracker_20261007';W=R/'work/jump_tracker_20261007';U=R/'outputs_amass/universal_tracker_20261006';py=sys.executable
subprocess.run([str(R/'.conda/bin/python'),str(W/'generate_expanded_training.py')],check=True)
subprocess.run([py,str(W/'prepare_expanded_training.py')],check=True)
subprocess.run([py,str(W/'plan_per_reference.py'),'--manifest',str(D/'expanded_training/native_manifest.json'),'--name','expanded_training40','--initial',str(D/'planning_initial.json')],check=True)
subprocess.run([py,str(W/'collect_pulse_teacher.py'),'--checkpoint',str(D/'backup/policy.pt'),'--manifest',str(D/'planning/expanded_training40/manifest.json'),'--name','expanded_teacher40','--workers','4'],check=True)
subprocess.run([py,str(W/'assess.py'),'--name','expanded_teacher40'],check=True)
rows=json.load(open(D/'general_evaluation/planned_teacher_train/audited_results.json'))+json.load(open(D/'general_evaluation/expanded_teacher40/audited_results.json'));manifest=D/'expanded_training/combined_teacher135.json';manifest.write_text(json.dumps(rows,indent=2))
subprocess.run([py,str(W/'train_residual_actor.py'),'--teacher',str(manifest),'--name','reference_residual_v6_expanded','--steps','20000','--reference-only','--hidden-width','256','--learning-rate','.0002','--cosine','--seed','710711','--extra-retention',str(D/'natural_retention/observations.npz')],check=True)
actor=D/'residual_training/reference_residual_v6_expanded/actor_19999.pt'
for split,mp in [('dev',D/'dev_jump_manifest.json'),('extended_dev',D/'extended_dev_jump_manifest.json'),('regression',U/'native_eleven_manifest.json'),('natural',U/'natural_test_motion/manifest.json')]:
 name='residual_v6_'+split;subprocess.run([py,str(W/'evaluate_actor.py'),'--checkpoint',str(actor),'--actor',str(actor),'--manifest',str(mp),'--name',name,'--workers','4'],check=True)
 if split=='natural':subprocess.run([py,str(R/'work/universal_tracker_20261006/assess_general_fidelity.py'),'--run',str(D/'general_evaluation'/name)],check=True)
 else:subprocess.run([py,str(W/'assess.py'),'--name',name],check=True)
subprocess.run([py,str(W/'evaluate_table_actor.py'),'--checkpoint',str(actor),'--actor',str(actor),'--name','residual_v6_table','--workers','3'],check=True)
