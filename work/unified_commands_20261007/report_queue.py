from pathlib import Path
import subprocess,time
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/unified_commands_20261007';W=R/'work/unified_commands_20261007'
while not (D/'general_evaluation/unified_final/assessment.json').exists():time.sleep(5)
for script in ['summarize.py','scene_summary.py','make_report.py']:
 with (D/(script+'.log')).open('w') as log:subprocess.run([str(R/'work/mjlab_stable_env/bin/python'),str(W/script)],cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
while not (D/'visuals/render_complete.json').exists():time.sleep(5)
with (D/'video_qc.log').open('w') as log:subprocess.run([str(R/'.conda/bin/python'),str(W/'video_qc.py')],cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
(D/'evaluation_complete.json').write_text('{}')
