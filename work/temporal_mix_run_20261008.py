from pathlib import Path
import subprocess,json,time
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/temporal_mix_20261008';W=R/'work/unified_commands_20261007';ap=R/'work/mjlab_stable_env/bin/python';name='temporal_mix_20261008';actor=D/'backup/tracker.pt'
while not (D/'generation_complete.json').exists():time.sleep(3)
steps=[['prepare_tracking.py','--name',name],['evaluate_tracker.py','--checkpoint',actor,'--actor',actor,'--manifest',R/'outputs_amass/unified_commands_20261007/generation'/name/'native_manifest.json','--name',name,'--workers','3']]
for i,args in enumerate(steps):
 with (D/f'step_{i}.log').open('w') as log:subprocess.run([str(ap),str(W/args[0])]+[str(x) for x in args[1:]],cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
(D/'physics_complete.json').write_text(json.dumps(dict(name=name,requests=6)))
