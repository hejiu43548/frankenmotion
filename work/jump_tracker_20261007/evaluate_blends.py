import sys,subprocess
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/jump_tracker_20261007';W=R/'work/jump_tracker_20261007';py=sys.executable
for weight in [50,75]:
 actor=D/'residual_training/blends'/f'actor_feedback{weight}.pt'
 for split in ['dev','extended_dev']:
  manifest=D/('dev_jump_manifest.json' if split=='dev' else 'extended_dev_jump_manifest.json');name=f'blend_{weight}_{split}'
  subprocess.run([py,str(W/'evaluate_actor.py'),'--checkpoint',str(actor),'--actor',str(actor),'--manifest',str(manifest),'--name',name,'--workers','4'],check=True)
  subprocess.run([py,str(W/'assess.py'),'--name',name],check=True)
