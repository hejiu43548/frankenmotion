"""Wait for completed snapshots and evaluate every fixed development scene."""
import os,sys,time,json,subprocess,hashlib,argparse
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/table_demo_20261005';sys.path.insert(0,str(R/'work'))
from unified_evaluation_lock_20261004 import evaluation_slot
p=argparse.ArgumentParser();p.add_argument('--training',default='interaction_v1');p.add_argument('--pid',type=int,required=True);p.add_argument('--steps',type=int,default=2000);p.add_argument('--development-folder',default='development_v2');p.add_argument('--reference',default='reference_lift');a=p.parse_args();folder=D/'training'/a.training;rows=json.loads((D/a.development_folder/'manifest.json').read_text());allresults=[]
for step in [s for s in [500,1000,1500] if s<a.steps]+[a.steps-1]:
 ckpt=folder/f'model_{step}.pt';ready=ckpt.with_suffix('.pt.ready.json')
 while not ready.exists():
  if not Path(f'/proc/{a.pid}').exists():raise RuntimeError(f'Training ended without snapshot {step}')
  time.sleep(10)
 assert hashlib.sha256(ckpt.read_bytes()).hexdigest()==json.loads(ready.read_text())['checkpoint_sha256'];name=f'{a.training}_{step:04d}';results=[]
 for row in rows:
  scene=Path(row['source']);dest=scene/name
  with evaluation_slot():
   with (scene/(name+'.log')).open('w') as log:subprocess.run([str(R/'work/mjlab_stable_env/bin/python'),str(R/'work/table_simulate_20261005.py'),'--scene',str(scene),'--name',name,'--reference',a.reference,'--checkpoint',str(ckpt)],cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
  result=json.loads((dest/'result.json').read_text());results.append(result);print(name,row['index'],result['success'],result['palm_target_error_m'],flush=True)
 record=dict(name=name,step=step,checkpoint=str(ckpt),checkpoint_sha256=hashlib.sha256(ckpt.read_bytes()).hexdigest(),successes=sum(r['success'] for r in results),planned=len(rows),complete=sum(r['physical_complete'] for r in results),mean_hand_error=sum(r['palm_target_error_m'] for r in results)/len(rows),mean_root_error=sum(r['goal_error_m'] for r in results)/len(rows),results=results);allresults.append(record);(folder/'development_results.json').write_text(json.dumps(allresults,indent=2));print('snapshot summary',name,record['successes'],record['mean_hand_error'],flush=True)
(folder/'development_complete.json').write_text(json.dumps(dict(candidates=len(allresults),planned_per_candidate=len(rows))))
