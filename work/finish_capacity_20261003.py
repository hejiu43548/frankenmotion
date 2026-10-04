from pathlib import Path
import json,os,time,subprocess
R=Path('/home/pku/frankenmotion');N=R/'outputs_amass/franken_improve_20261003';W=R/'work';pid=357216
cases=[('integrated_v3_beyondmimic_capacity256',240,['assess_integrated_v3_capacity','plot_integrated_v3_capacity']),('novel_prompt_v3_beyondmimic_capacity256',30,['assess_novel_prompt_v3_capacity']),('beyondmimic_confirmation_capacity256',880,['recheck_trajectories_capacity','assess_confirmation_capacity','plot_confirmation_capacity'])]
for folder,count,scripts in cases:
 path=N/folder/'results.json'
 while not path.exists():
  os.kill(pid,0);time.sleep(5)
 assert len(json.loads(path.read_text()))==count
 for name in scripts:
  print('START',name,flush=True);env=dict(os.environ,PYTHONPATH=str(R/'outputs_amass/g1_command_diagnosis_20261002_2346/plot_deps'));subprocess.run([str(W/'g1_sim_env/bin/python'),str(W/(name+'_20261003.py'))],cwd=R,env=env,check=True);print('DONE',name,flush=True)
print('Corrected data and plots complete; report impact versus historical and verify archive',flush=True)
