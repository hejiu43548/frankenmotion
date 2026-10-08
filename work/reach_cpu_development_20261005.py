import argparse,json,subprocess
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/reach_demo_20261005';p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--name',required=True);p.add_argument('--checkpoint',required=True);a=p.parse_args();actor=D/(a.name+'_actor.pt');py=R/'work/mjlab_stable_env/bin/python'
def run(name,args):subprocess.run([str(py),str(R/'work'/name)]+args,check=True)
if not actor.exists():run('unified_export_actor_20261004.py',['--checkpoint',a.checkpoint,'--output',str(actor)])
folder=D/a.folder;rows=json.loads((folder/'manifest.json').read_text());rr=[]
for row in rows:
 s=Path(row['source']);out=s/(a.name+'_cpu');run('reach_cpu_evaluate_20261005.py',['--run',str(s/a.name),'--actor',str(actor),'--checkpoint',a.checkpoint,'--output',str(out)]+(['--check-observations'] if row['index']==0 else []));rr.append(json.loads((out/'result.json').read_text()));print('CPU_RESULT',row['index'],rr[-1]['success'],flush=True)
(folder/(a.name+'_cpu_summary.json')).write_text(json.dumps(dict(count=len(rr),successes=sum(r['success'] for r in rr),results=rr),indent=2))
