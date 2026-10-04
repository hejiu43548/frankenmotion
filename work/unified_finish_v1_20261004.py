from pathlib import Path
import os,time,json,subprocess
R=Path('/home/pku/frankenmotion');U=R/'outputs_amass/franken_unified_20261004';pid=373322
while not (U/'training/joint_v1/complete.json').exists():
 os.kill(pid,0);time.sleep(5)
assert json.loads((U/'training/joint_v1/complete.json').read_text())['iterations']==3000
assert (U/'development_validation/reference_manifest.json').exists(),'Reference preparation failed or incomplete; inspect separately'
py=str(R/'work/mjlab_stable_env/bin/python');subprocess.run([py,str(R/'work/unified_evaluate_20261004.py'),'--checkpoint',str(U/'training/joint_v1/model_2999.pt'),'--name','joint_v1_validation'],cwd=R,check=True)
subprocess.run([str(R/'work/g1_sim_env/bin/python'),str(R/'work/unified_assess_20261004.py'),'--name','joint_v1_validation'],cwd=R,check=True)
