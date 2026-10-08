import sys,json,argparse,subprocess,hashlib
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/table_demo_20261005';sys.path.insert(0,str(R/'work'))
from unified_evaluation_lock_20261004 import evaluation_slot
p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--name',required=True);p.add_argument('--reference',default='reference_contact');p.add_argument('--fresh-initial-observation',action='store_true');a=p.parse_args();folder=Path(a.folder).resolve();results=[];rows=json.loads((folder/'manifest.json').read_text())
for row in rows:
 scene=Path(row['source']);dest=scene/a.name
 assert not dest.exists(),dest
 with evaluation_slot():
  with (scene/(a.name+'.log')).open('w') as log:subprocess.run([str(R/'work/mjlab_stable_env/bin/python'),str(R/'work/turn_simulate_20261005.py'),'--scene',str(scene),'--name',a.name,'--reference',a.reference,'--checkpoint',str(Path(a.checkpoint).resolve())]+(['--fresh-initial-observation'] if a.fresh_initial_observation else []),cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
 assert 'overflow' not in (scene/(a.name+'.log')).read_text().lower()
 result=json.loads((dest/'result.json').read_text());results.append(result);summary=dict(name=a.name,checkpoint=str(Path(a.checkpoint).resolve()),checkpoint_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest(),successes=sum(r['success'] for r in results),processed=len(results),planned=len(rows),complete=sum(r['physical_complete'] for r in results),mean_reach_error_human_m=sum(r['command_metrics']['command_error_m'] for r in results if r['command_metrics']['command_error_m'] is not None)/max(1,sum(r['command_metrics']['command_error_m'] is not None for r in results)),mean_hold_root_error_m=sum(r['command_metrics']['mean_hold_root_error_m'] for r in results if r['command_metrics']['mean_hold_root_error_m'] is not None)/max(1,sum(r['command_metrics']['mean_hold_root_error_m'] is not None for r in results)),results=results)
 (folder/(a.name+'_summary.json')).write_text(json.dumps(summary,indent=2));print(row['index'],result['success'],{k:result['command_metrics'][k] for k in ['actual_hold_wrist_forward_human_equiv_m','longest_hold_contact_s','hand_lowering_m','signed_exit_distance_m']},flush=True)
print('COMPLETE',summary['successes'],len(rows),summary['mean_reach_error_human_m'],flush=True)
