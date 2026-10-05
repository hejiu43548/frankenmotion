import sys,json,argparse,subprocess,hashlib
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/table_demo_20261005';sys.path.insert(0,str(R/'work'))
from unified_evaluation_lock_20261004 import evaluation_slot
p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--name',required=True);p.add_argument('--reference',default='reference_contact');p.add_argument('--fresh-initial-observation',action='store_true');a=p.parse_args();folder=Path(a.folder).resolve();results=[];rows=json.loads((folder/'manifest.json').read_text())
for row in rows:
 scene=Path(row['source']);dest=scene/a.name
 assert not dest.exists(),dest
 with evaluation_slot():
  with (scene/(a.name+'.log')).open('w') as log:subprocess.run([str(R/'work/mjlab_stable_env/bin/python'),str(R/'work/table_simulate_20261005.py'),'--scene',str(scene),'--name',a.name,'--reference',a.reference,'--checkpoint',str(Path(a.checkpoint).resolve())]+(['--fresh-initial-observation'] if a.fresh_initial_observation else []),cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
 assert 'overflow' not in (scene/(a.name+'.log')).read_text().lower()
 result=json.loads((dest/'result.json').read_text());results.append(result);summary=dict(name=a.name,checkpoint=str(Path(a.checkpoint).resolve()),checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),successes=sum(r['success'] for r in results),processed=len(results),planned=len(rows),complete=sum(r['physical_complete'] for r in results),mean_hand_error=sum(r['palm_target_error_m'] for r in results)/len(results),mean_root_error=sum(r['goal_error_m'] for r in results)/len(results),results=results)
 (folder/(a.name+'_summary.json')).write_text(json.dumps(summary,indent=2));print(row['index'],result['success'],result['goal_error_m'],result['palm_target_error_m'],result['longest_hand_contact_s'],flush=True)
print('COMPLETE',summary['successes'],len(rows),summary['mean_hand_error'],flush=True)
