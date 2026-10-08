from pathlib import Path
import subprocess,time,argparse
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/unified_commands_20261007';W=R/'work/unified_commands_20261007';py=R/'.conda/bin/python';ap=R/'work/mjlab_stable_env/bin/python';actor=D/'backup/tracker.pt';name='unified_final'
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);a=p.parse_args()
def run(exe,args,log):
 with (D/log).open('w') as f:subprocess.run([str(x) for x in [exe]+args],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
run(py,[W/'evaluate_generation.py','--name',name,'--checkpoint',a.checkpoint,'--full','--split','final'],name+'_0.log')
while not (D/'generation/baseline_final/complete.json').exists():time.sleep(5)
for i,(exe,args) in enumerate([(ap,[W/'compare_generation.py','--name',name,'--baseline','baseline_final']),(ap,[W/'prepare_tracking.py','--name',name]),(ap,[W/'evaluate_tracker.py','--checkpoint',actor,'--actor',actor,'--manifest',D/'generation'/name/'native_manifest.json','--name',name,'--workers',4]),(ap,[W/'assess_physics.py','--name',name])],1):run(exe,args,f'{name}_{i}.log')
