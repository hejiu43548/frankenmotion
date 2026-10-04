from pathlib import Path
import os,time,json,subprocess
R=Path('/home/pku/frankenmotion');U=R/'outputs_amass/franken_unified_20261004'
for name,pid,preview in [('joint_v2_plain',383029,False),('joint_v2_preview',383263,True)]:
 while not (U/'training'/name/'complete.json').exists():
  os.kill(pid,0);time.sleep(5)
 assert json.loads((U/'training'/name/'complete.json').read_text())['iterations']==3000
 cmd=[str(R/'work/mjlab_stable_env/bin/python'),str(R/'work/unified_evaluate_20261004.py'),'--checkpoint',str(U/'training'/name/'model_2999.pt'),'--name',name+'_validation']
 if preview:cmd.append('--preview')
 subprocess.run(cmd,cwd=R,check=True);subprocess.run([str(R/'work/g1_sim_env/bin/python'),str(R/'work/unified_assess_20261004.py'),'--name',name+'_validation'],cwd=R,check=True)
