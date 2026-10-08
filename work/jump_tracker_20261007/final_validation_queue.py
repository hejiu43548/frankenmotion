import sys,subprocess
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/jump_tracker_20261007';W=R/'work/jump_tracker_20261007';py=sys.executable
def evaluate(script,actor,name,profile=None):
 cmd=[py,str(W/script),'--checkpoint',str(actor),'--actor',str(actor),'--manifest',str(D/'dev_jump_manifest.json'),'--name',name,'--workers','4']
 if profile:cmd+=['--profile',profile]
 subprocess.run(cmd,check=True)
 subprocess.run([py,str(W/'assess.py'),'--name',name],check=True)
v3=D/'residual_training/reference_residual_v3_wide/actor_14999.pt';v4=D/'residual_training/reference_residual_v4_broad/actor_14999.pt'
evaluate('evaluate_predictive.py',v3,'predictive_v3_mass_mismatch','delay_predict_mass_1p1')
for profile in ['delay_20ms','mass_1p1','friction_0p6','lateral_push_40N']:
 evaluate('evaluate_robust_actor.py',v4,'robust_v4_'+profile,profile)
evaluate('evaluate_predictive.py',v4,'predictive_v4','delay_predict')
evaluate('evaluate_predictive.py',v4,'predictive_v4_mass_mismatch','delay_predict_mass_1p1')
subprocess.run([py,str(W/'evaluate_table_actor.py'),'--checkpoint',str(v4),'--actor',str(v4),'--name','residual_v4_table','--workers','3'],check=True)
