from pathlib import Path
import subprocess
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/unified_commands_20261007';W=R/'work/unified_commands_20261007';py=R/'.conda/bin/python';ap=R/'work/mjlab_stable_env/bin/python';actor=D/'backup/tracker.pt'
def run(args,log):
 with (D/log).open('w') as f:subprocess.run([str(x) for x in args],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
name='feature_v3_dev';ck=D/'training/feature_v3_wide/model_30000.pt'
run([py,W/'evaluate_generation.py','--name',name,'--checkpoint',ck],name+'.log')
run([ap,W/'compare_generation.py','--name',name],name+'_comparison.log')
run([ap,W/'prepare_tracking.py','--name',name],name+'_retarget.log')
run([ap,W/'evaluate_tracker.py','--checkpoint',actor,'--actor',actor,'--manifest',D/'generation'/name/'native_manifest.json','--name',name,'--workers',4],name+'_tracking.log')
run([ap,W/'assess_physics.py','--name',name],name+'_assessment.log')
