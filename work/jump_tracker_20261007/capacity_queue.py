"""Last two architecture probes on expanded training only; same held-out validation."""
import sys,json,subprocess
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/jump_tracker_20261007';W=R/'work/jump_tracker_20261007';U=R/'outputs_amass/universal_tracker_20261006';py=sys.executable
for version,width,reference in [('v7_feedback',256,False),('v8_capacity',512,True)]:
 name='residual_'+version;steps=25000;cmd=[py,str(W/'train_residual_actor.py'),'--teacher',str(D/'expanded_training/combined_teacher135.json'),'--name',name,'--steps',str(steps),'--hidden-width',str(width),'--learning-rate','.0002','--cosine','--seed','710711','--extra-retention',str(D/'natural_retention/observations.npz')]
 if reference:cmd.append('--reference-only')
 subprocess.run(cmd,check=True);actor=D/'residual_training'/name/f'actor_{steps-1}.pt'
 for split,mp in [('dev',D/'dev_jump_manifest.json'),('extended_dev',D/'extended_dev_jump_manifest.json'),('regression',U/'native_eleven_manifest.json'),('natural',U/'natural_test_motion/manifest.json')]:
  tag=name+'_'+split;subprocess.run([py,str(W/'evaluate_actor.py'),'--checkpoint',str(actor),'--actor',str(actor),'--manifest',str(mp),'--name',tag,'--workers','4'],check=True)
  if split=='natural':subprocess.run([py,str(R/'work/universal_tracker_20261006/assess_general_fidelity.py'),'--run',str(D/'general_evaluation'/tag)],check=True)
  else:subprocess.run([py,str(W/'assess.py'),'--name',tag],check=True)
 subprocess.run([py,str(W/'evaluate_table_actor.py'),'--checkpoint',str(actor),'--actor',str(actor),'--name',name+'_table','--workers','3'],check=True)
