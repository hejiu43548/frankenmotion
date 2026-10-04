"""Wait for the already-running pipeline, then audit and plot its completed data."""
import os,time,json,subprocess
from pathlib import Path
ROOT=Path('/home/pku/frankenmotion');NEW=ROOT/'outputs_amass/franken_improve_20261003';W=ROOT/'work';pid=335757
while True:
 try:os.kill(pid,0)
 except ProcessLookupError:break
 time.sleep(5)
for path,n in [(NEW/'integrated_v3_sonic/results.json',880),(NEW/'confirmation/integrated_v3_control/results.json',240),(NEW/'integrated_v3_beyondmimic/results.json',240)]:assert len(json.loads(path.read_text()))==n,(path,n)
for name in ['assess_integrated_v3','plot_integrated_v3']:
 print('START',name,flush=True);env=dict(os.environ,PYTHONPATH=str(ROOT/'outputs_amass/g1_command_diagnosis_20261002_2346/plot_deps'));subprocess.run([str(W/'g1_sim_env/bin/python'),str(W/(name+'_20261003.py'))],cwd=ROOT,env=env,check=True);print('DONE',name,flush=True)
print('V3 data and plots complete; manual review, report and package still required',flush=True)
