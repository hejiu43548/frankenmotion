from pathlib import Path
import subprocess
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/universal_tracker_20261006';W=R/'work/universal_tracker_20261006';py=str(R/'work/mjlab_stable_env/bin/python');ck=str(D/'training/joint_v3_broad_init/model_1000.pt')
for name,manifest,split in [('refined_v3_1000_flat',D/'generated_refined_motion/manifest.json',None),('natural_v3_1000_flat',D/'natural_extended_motion/manifest.json','val'),('official_v3_1000_flat',D/'generated_official_motion/manifest.json',None)]:
 cmd=[py,str(W/'evaluate_cpu_general.py'),'--checkpoint',ck,'--manifest',str(manifest),'--name',name]
 if split:cmd.extend(['--split',split])
 with (D/(name+'.log')).open('w') as log:subprocess.run(cmd,cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
 print(name,'complete',flush=True)
