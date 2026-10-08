import sys,subprocess
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/jump_tracker_20261007';W=R/'work/jump_tracker_20261007';U=R/'outputs_amass/universal_tracker_20261006';py=sys.executable;step=int(sys.argv[1]);checkpoint=D/f'training/envelope_shared_4000/model_{step}.pt';actor=D/f'general_evaluation/envelope_shared_4000_{step}_jump/actor.pt'
for split,manifest in [('extended_dev',D/'extended_dev_jump_manifest.json'),('regression',U/'native_eleven_manifest.json'),('natural',U/'natural_test_motion/manifest.json')]:
 name=f'envelope_{step}_'+split;subprocess.run([py,str(W/'evaluate_actor.py'),'--checkpoint',str(checkpoint),'--actor',str(actor),'--manifest',str(manifest),'--name',name,'--workers','4'],check=True)
 if split=='natural':subprocess.run([py,str(R/'work/universal_tracker_20261006/assess_general_fidelity.py'),'--run',str(D/'general_evaluation'/name)],check=True)
 else:subprocess.run([py,str(W/'assess.py'),'--name',name],check=True)
subprocess.run([py,str(W/'evaluate_table_actor.py'),'--checkpoint',str(checkpoint),'--actor',str(actor),'--name',f'envelope_{step}_table','--workers','3'],check=True)
