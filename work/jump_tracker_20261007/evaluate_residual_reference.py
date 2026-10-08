import sys,subprocess
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/jump_tracker_20261007';W=R/'work/jump_tracker_20261007';py=sys.executable
for step in [499,1499,9999]:
 actor=D/'residual_training/reference_residual_v2'/f'actor_{step}.pt';name=f'reference_residual_v2_{step}'
 subprocess.run([py,str(W/'evaluate_actor.py'),'--checkpoint',str(actor),'--actor',str(actor),'--manifest',str(D/'dev_jump_manifest.json'),'--name',name,'--workers','4'],check=True)
 subprocess.run([py,str(W/'assess.py'),'--name',name],check=True)
