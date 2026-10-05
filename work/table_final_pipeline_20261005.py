"""Freeze development-selected shared controller before random final scenes exist."""
import sys,time,json,hashlib,shutil,subprocess,datetime,argparse
import numpy as np
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/table_demo_20261005';U=R/'outputs_amass/franken_unified_20261004'
p=argparse.ArgumentParser();p.add_argument('--table-watcher-pid',required=True,type=int);p.add_argument('--legacy-watcher-pid',required=True,type=int);a=p.parse_args()
read=lambda x:json.loads(Path(x).read_text());sha=lambda x:hashlib.sha256(Path(x).read_bytes()).hexdigest()
def call(runtime,script,args):
 print('START',script,args,flush=True);subprocess.run([str(R/runtime),str(R/'work'/script),*args],cwd=R,check=True)
for filename,pid in [('development_complete.json',a.table_watcher_pid),('legacy_complete.json',a.legacy_watcher_pid)]:
 while not (D/'training/retention_v2'/filename).exists():
  if not Path(f'/proc/{pid}').exists():raise RuntimeError('Watcher exited without '+filename)
  time.sleep(10)
assert not (D/'final_test').exists() and not (D/'frozen').exists()
original=U/'frozen_unified/policy.pt';original_table=read(D/'development_v3/original_tracker_summary.json');original_legacy=read(U/'evaluation/joint_v5_root_short_validation/audit.json');v1_table=read(D/'development_v3/interaction_v1_1000_summary.json');v1_legacy=read(U/'evaluation/table_interaction_1000_20261005/audit.json');candidates=[]
def append(name,checkpoint,table,legacy):
 assert table['planned']==6 and table.get('processed',6)==6 and len(table['results'])==6
 assert legacy['raw_max_error']<1e-8 and legacy['overflow_warnings']==0 and legacy['aggregate']['requests']==110
 assert sha(checkpoint)==legacy['single_checkpoint_sha256']==table['checkpoint_sha256'];q=legacy['aggregate'];eligible=table['successes']==6 and q['complete']>=109 and q['event']>=104 and q['joint']>=45 and q['macro_semantic_E_all']<=.185
 candidates.append(dict(name=name,checkpoint=str(checkpoint),sha256=sha(checkpoint),table_successes=table['successes'],table_mean_hand_error=table['mean_hand_error'],legacy=q,eligible=eligible))
append('original_shared_tracker',original,original_table,original_legacy);append('interaction_v1_step1000',D/'training/interaction_v1/model_1000.pt',v1_table,v1_legacy)
tables=read(D/'training/retention_v2/development_results.json');legacies=read(D/'training/retention_v2/legacy_results.json')
for table in tables:
 legacy=next(x['audit'] for x in legacies if x['step']==table['step']);append(table['name'],Path(table['checkpoint']),table,legacy)
selected=min([x for x in candidates if x['eligible']],key=lambda x:x['table_mean_hand_error']);(D/'development_selection.json').write_text(json.dumps(dict(candidates=candidates,selected=selected),indent=2));print('SELECTED',selected,flush=True)
# Observation-cache correction is independently checked on development before freeze.
for name,checkpoint in [('selected_fresh',selected['checkpoint']),('original_fresh',str(original))]:
 call('work/g1_sim_env/bin/python','table_evaluate_scenes_20261005.py',['--folder',str(D/'development_v3'),'--checkpoint',checkpoint,'--name',name,'--fresh-initial-observation'])
fresh=read(D/'development_v3/selected_fresh_summary.json');fallback=read(D/'development_v3/original_fresh_summary.json');reason='Development eligibility and minimum hand error, revalidated with fresh initial observations'
if fresh['successes']<6:
 assert fallback['successes']==6,'Both selected and original failed fresh development; inspect before final data'
 selected=candidates[0];reason='Conservative original-policy fallback after fresh initial-observation check'
for row in read(D/'development_v3/manifest.json'):
 for name in ['selected_fresh','original_fresh']:call('work/mjlab_stable_env/bin/python','table_audit_rollout_20261005.py',['--run',str(Path(row['source'])/name)])
out=D/'frozen';out.mkdir();shutil.copy2(selected['checkpoint'],out/'policy.pt');shutil.copy2(D/'goal_v2/best.pt',out/'goal_adapter.pt')
protocol=dict(frozen_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),selected=selected,candidates=candidates,selection_reason=reason,tracker_sha256=sha(out/'policy.pt'),goal_sha256=sha(out/'goal_adapter.pt'),task_routing=False,known_simulator_table_pose=True,final_scene_count=32,final_scene_seed=55005000,generation_refinement_iterations=3,contact_preload_m=.015,initial_observation='Explicit observation-manager cache reset after physical initialization',reference_pipeline='Learned goal-conditioned generation, up to three same-noise command-space corrections; GMR preserves resulting approach path; smooth stand and scene-aware right-arm IK lift/extend/lower/hold.',scope='Front sector, stop radius 0.85–1.75m, direction ±0.5rad, 0.8m-high table. One full-body actor across every scene. No visual perception, general obstacle avoidance or hardware validation.',source_sha256={x.name:sha(x) for x in sorted((R/'work').glob('table_*_20261005.py'))})
(out/'protocol.json').write_text(json.dumps(protocol,indent=2));call('work/mjlab_stable_env/bin/python','unified_export_actor_20261004.py',['--checkpoint',str(out/'policy.pt'),'--output',str(out/'actor.pt')])
call('.conda/bin/python','table_generate_scenes_20261005.py',['--weight',str(out/'goal_adapter.pt'),'--name','final_test','--count','32','--seed','55005000','--all-random','--refine-iterations','3'])
call('work/g1_sim_env/bin/python','table_prepare_reference_20261005.py',['--folder',str(D/'final_test'),'--tag','reference_contact','--lift','--hold-extra','60','--contact-preload','.015'])
rows=read(D/'final_test/manifest.json');assert len(rows)==32 and {r['seed'] for r in rows}==set(range(55005000,55005032));assert all(r['weight_sha256']==protocol['goal_sha256'] for r in rows)
seen=set()
for split in ['training_v2','development_v1','development_v2','development_v3']:seen.update(x['seed'] for x in read(D/split/'manifest.json'))
seen.update(range(51005000,51007401));seen.update([52005000,52005001]);seen.update(x['seed'] for x in read(D/'goal_command_audit/results.json'));assert not seen.intersection({r['seed'] for r in rows})
rng=np.random.default_rng(55005000)
for row in rows:
 expected_distance=float(rng.uniform(.85,1.75));expected_angle=float(rng.uniform(-.5,.5));assert abs(row['distance_robot_m']-expected_distance)<1e-12 and abs(row['direction_rad']-expected_angle)<1e-12
 assert (out/'protocol.json').stat().st_mtime<(Path(row['source'])/'human_walk.npz').stat().st_mtime
(D/'final_test/cohort_audit.json').write_text(json.dumps(dict(planned=32,seeds_disjoint_from_training_and_development=True,all_random=True,freeze_precedes_generation=(out/'protocol.json').stat().st_mtime<(D/'final_test/manifest.json').stat().st_mtime,manifest_sha256=sha(D/'final_test/manifest.json')),indent=2))
methods=[('unified',out/'policy.pt')]
if sha(original)!=protocol['tracker_sha256']:methods.append(('original_tracker',original))
for name,checkpoint in methods:
 call('work/g1_sim_env/bin/python','table_evaluate_scenes_20261005.py',['--folder',str(D/'final_test'),'--checkpoint',str(checkpoint),'--name',name,'--fresh-initial-observation'])
 for row in rows:call('work/mjlab_stable_env/bin/python','table_audit_rollout_20261005.py',['--run',str(Path(row['source'])/name)])
(D/'final_pipeline_complete.json').write_text(json.dumps(dict(methods=[n for n,_ in methods],selected=selected['name'],final_count=32,original_aliases_unified=len(methods)==1),indent=2));print('FINAL PHYSICAL TESTS AND AUDITS COMPLETE',flush=True)
