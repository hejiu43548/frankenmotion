import sys,subprocess
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/jump_tracker_20261007';W=R/'work/jump_tracker_20261007';py=sys.executable
models=[('baseline',D/'backup/policy.pt',D/'backup/actor.pt'),('reference_v3',D/'residual_training/reference_residual_v3_wide/actor_14999.pt',D/'residual_training/reference_residual_v3_wide/actor_14999.pt'),('blend75',D/'residual_training/blends/actor_feedback75.pt',D/'residual_training/blends/actor_feedback75.pt')]
for label,checkpoint,actor in models:
 for profile in ['friction_0p6','mass_1p1','delay_20ms','lateral_push_40N']:
  name=f'robust_{label}_{profile}'
  subprocess.run([py,str(W/'evaluate_robust_actor.py'),'--checkpoint',str(checkpoint),'--actor',str(actor),'--profile',profile,'--manifest',str(D/'dev_jump_manifest.json'),'--name',name,'--workers','4'],check=True)
  subprocess.run([py,str(W/'assess.py'),'--name',name],check=True)
