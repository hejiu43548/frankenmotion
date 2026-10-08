import json,time,subprocess
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/unified_generator_20261007';W=R/'work/unified_generator_20261007';tasks=['raise_hand','reach','strike','wave','turn','sidestep','back_walk','kick','jump','lean','walk']
def run(cmd,log):
 with (D/log).open('w') as f:subprocess.run([str(x) for x in cmd],stdout=f,stderr=subprocess.STDOUT,check=True,cwd=R)
while not (D/'generation/unified_final/audit.json').exists():time.sleep(10)
run([R/'.conda/bin/python',W/'render_human_compare.py','--name','unified_final','--baseline','teacher_final','--tasks']+tasks,'final_human_render.log')
while not (D/'final_evaluation_complete.json').exists():time.sleep(10)
run([R/'.conda/bin/python',W/'summarize_final.py'],'final_summary.log')
base=D/'general_evaluation/unified_final';out=D/'visuals/g1_unified_final';out.mkdir(exist_ok=True)
for task in tasks:
 for ci in [0,2,4]:
  run([R/'work/mjlab_stable_env/bin/python',R/'work/universal_tracker_20261006/render_general.py','--run',base/(task+f'_p0_s0_c{ci}'),'--output',out/(task+f'_c{ci}'),'--follow'],f'render_g1_{task}_c{ci}.log')
(D/'render_final_complete.json').write_text(json.dumps(dict(tasks=tasks,scope='33 fixed human paired clips and33 fixed G1 retarget-versus-physics clips. First prompt/noise, low/mid/high, not selected for success.')))
