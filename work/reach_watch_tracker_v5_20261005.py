import time,json,subprocess
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/reach_demo_20261005';records=[]
for step in [500,1000,1500,1999]:
 ck=D/f'training/tracker_v5/model_{step}.pt'
 while not ck.with_suffix('.pt.ready.json').exists():time.sleep(5)
 name=f'tracker_v5_{step}'
 with (R/f'work/reach_eval_{name}.log').open('w') as log:subprocess.run([str(R/'work/g1_sim_env/bin/python'),str(R/'work/reach_evaluate_20261005.py'),'--folder',str(D/'development_v7'),'--checkpoint',str(ck),'--name',name,'--fresh-initial-observation'],cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
 summary=json.loads((D/'development_v7'/f'{name}_summary.json').read_text());rr=[r['command_metrics'] for r in summary['results']];record=dict(step=step,successes=sum(r['success'] for r in rr),complete=sum(r['physical_complete'] for r in rr),cases=rr);records.append(record);(D/'tracker_v5_development_results.json').write_text(json.dumps(records,indent=2));print(step,record['successes'],record['complete'],[(r['command_human_equiv_m'],r['actual_hold_wrist_forward_human_equiv_m'],r['longest_table_contact_s'],r['hand_lowering_m'],r['signed_exit_distance_m']) for r in rr],flush=True)
(D/'tracker_v5_watch_complete.json').write_text(json.dumps(dict(count=len(records))))
