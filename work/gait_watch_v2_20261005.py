import json,time,subprocess
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/gait_demo_20261005';runtime=R/'work/mjlab_stable_env/bin/python'
reports=[]
for step in [500,1000,1500,2000,2499]:
 ck=D/f'training/sole_v2/model_{step}.pt'
 while not ck.with_suffix('.pt.ready.json').exists():time.sleep(15)
 for kind in ['timed','contact_v2']:
  name=f'sole_v2_{step}';folder=D/f'probe_{kind}'
  subprocess.run([str(R/'work/g1_sim_env/bin/python'),str(R/'work/gait_evaluate_20261005.py'),'--folder',str(folder),'--checkpoint',str(ck),'--name',name,'--fresh-initial-observation'],cwd=R,check=True)
  s=json.loads((folder/(name+'_summary.json')).read_text());g=[]
  for r in s['results']:
   run=Path(r['scene']['source'])/name;subprocess.run([str(runtime),str(R/'work/gait_metrics_20261005.py'),'--run',str(run)],cwd=R,check=True);g.append(json.loads((run/'gait_metrics.json').read_text()))
  reports.append(dict(step=step,kind=kind,summary=s,gait=g));(D/'sole_probe_results.json').write_text(json.dumps(reports,indent=2));print('PROBE',step,kind,s['successes'],[r['knee_amplitude_ratio'] for r in g],flush=True)
(D/'sole_probe_complete.json').write_text(json.dumps(dict(complete=True)))
