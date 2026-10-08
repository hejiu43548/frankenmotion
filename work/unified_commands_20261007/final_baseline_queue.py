from pathlib import Path
import subprocess
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/unified_commands_20261007';W=R/'work/unified_commands_20261007';py=R/'.conda/bin/python';ap=R/'work/mjlab_stable_env/bin/python';actor=D/'backup/tracker.pt';name='baseline_final'
steps=[(py,[W/'evaluate_generation.py','--name',name,'--teachers','--split','final']),(ap,[W/'compare_generation.py','--name',name,'--baseline',name]),(ap,[W/'prepare_tracking.py','--name',name]),(ap,[W/'evaluate_tracker.py','--checkpoint',actor,'--actor',actor,'--manifest',D/'generation'/name/'native_manifest.json','--name',name,'--workers',4]),(ap,[W/'assess_physics.py','--name',name])]
for i,(exe,args) in enumerate(steps):
 with (D/f'{name}_{i}.log').open('w') as f:subprocess.run([str(x) for x in [exe]+args],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
