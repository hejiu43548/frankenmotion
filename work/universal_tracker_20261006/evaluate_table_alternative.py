"""Evaluate a fixed candidate on unchanged historical development table scenes."""
import argparse,json,subprocess,shutil,hashlib
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/universal_tracker_20261006'
p=argparse.ArgumentParser();p.add_argument('--scope',default='Declared alternative references; development diagnostic only. One fixed actor for all scenes.');p.add_argument('--scenes-folder',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--name',required=True);p.add_argument('--workers',type=int,default=2);a=p.parse_args()
out=D/'table_evaluation'/a.name;out.mkdir(parents=True,exist_ok=False)
py=str(R/'work/mjlab_stable_env/bin/python');actor=out/'actor.pt'
with (out/'export.log').open('w') as log:
    subprocess.run([py,str(R/'work/unified_export_actor_20261004.py'),'--checkpoint',a.checkpoint,'--output',str(actor)],stdout=log,stderr=subprocess.STDOUT,check=True)
metadata=json.loads(actor.with_suffix('.json').read_text())
source=Path(a.scenes_folder)
rows=json.loads((source/'manifest.json').read_text())

def one(row):
    src=Path(row['source']);scene=out/f'scene_{row["index"]:03d}';scene.mkdir()
    shutil.copy2(src/'reference_contact.json',scene/'reference_contact.json')
    shutil.copytree(src/'reference_input',scene/'reference_input')
    contract=scene/'reference_input/inference_contract.json';c=json.loads(contract.read_text());c['preview_offsets']=metadata['preview_offsets'];contract.write_text(json.dumps(c,indent=2))
    run=scene/'rollout';cmd=[py,str(R/'work/turn_cpu_anchor_evaluate_20261005.py'),'--run',str(scene/'reference_input'),'--actor',str(actor),'--checkpoint',a.checkpoint,'--output',str(run)]
    with (scene/'execution.log').open('w') as log:result=subprocess.run(cmd,stdout=log,stderr=subprocess.STDOUT)
    if result.returncode:return dict(index=row['index'],error='runtime_or_metric_failure',returncode=result.returncode,log=str(scene/'execution.log'))
    record=json.loads((run/'result.json').read_text())
    for segment in ['walk','away']:
        with (scene/(segment+'_jitter.log')).open('w') as log:subprocess.run([py,str(R/'work/turn_jitter_20261005.py'),'--run',str(run),'--segment',segment],stdout=log,stderr=subprocess.STDOUT)
    record['run']=str(run);return record

results=[]
with ThreadPoolExecutor(max_workers=a.workers) as pool:
    for result in pool.map(one,rows):
        results.append(result)
        summary=dict(planned=len(rows),processed=len(results),complete=sum(r.get('physical_complete',False) for r in results),success=sum(r.get('success',False) for r in results),checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),scope=a.scope,results=results)
        (out/'summary.json').write_text(json.dumps(summary,indent=2));print(len(results),summary['complete'],summary['success'],flush=True)
