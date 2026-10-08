from pathlib import Path
import subprocess,time,json
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/unified_commands_20261007';W=R/'work/unified_commands_20261007';py=R/'.conda/bin/python';ap=R/'work/mjlab_stable_env/bin/python';tasks=['raise_hand','reach','strike','wave','turn','sidestep','back_walk','kick','jump','lean','walk'];extras=['walk_endpoint','place_hold_retract','back_departure','side_departure','root_profile','turn_endpoint']
def run(exe,args,log):
 with (D/log).open('w') as f:subprocess.run([str(x) for x in [exe]+args],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
while not (D/'generation/unified_final/comparison.json').exists():time.sleep(5)
run(py,[W/'render_human_compare.py','--name','unified_final','--baseline','baseline_final','--tasks']+tasks+extras,'human_render.log')
while not (D/'general_evaluation/unified_final/assessment.json').exists():time.sleep(5)
selection={t:[2] for t in tasks};selection.update({t:[0,2,4] for t in ['walk','jump','kick']});selection.update(dict(walk_endpoint=[0,8],place_hold_retract=[1,7],back_departure=[1],side_departure=[1],root_profile=[0],turn_endpoint=[0,2]))
for kind,indices in selection.items():
 for ci in indices:
  run(ap,[R/'work/universal_tracker_20261006/render_general.py','--run',D/'general_evaluation/unified_final'/f'{kind}_p0_s0_c{ci}','--output',D/'visuals/g1_unified_final'/f'{kind}_c{ci}','--follow'],f'render_g1_{kind}_{ci}.log')
(D/'visuals/render_complete.json').write_text(json.dumps(dict(human='17 contexts, prespecified command indices, first prompt/noise',g1_selection=selection,success_filter=False),indent=2))
