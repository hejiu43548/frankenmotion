from pathlib import Path
import subprocess
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/unified_commands_20261007';py=R/'work/mjlab_stable_env/bin/python'
for name,ids in [('unified_final',[0,1,2,3]),('baseline_final',[0,1])]:
 for i in ids:
  folder=D/'visuals'/('table' if name=='unified_final' else 'table_baseline');folder.mkdir(parents=True,exist_ok=True)
  with (D/f'render_table_{name}_{i}.log').open('w') as log:subprocess.run([str(py),str(R/'work/turn_render_20261005.py'),'--run',str(D/'table'/name/f'composed/scene_{i:03d}/shared_tracker'),'--output',str(folder/f'scene_{i:03d}.mp4'),'--label','ONE generator control | Reach, turn & depart' if name=='unified_final' else 'Original separate controls | Same shared tracker'],cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
