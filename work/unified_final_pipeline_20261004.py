"""Freeze global development winner, then run a never-before-created final cohort."""
import concurrent.futures,datetime,hashlib,json,os,subprocess,time
from pathlib import Path
R=Path('/home/pku/frankenmotion');U=R/'outputs_amass/franken_unified_20261004';CPU=R/'work/g1_sim_env/bin/python';GPU=R/'work/mjlab_stable_env/bin/python';GEN=R/'.conda/bin/python'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
def run(script,args=(),python=CPU,env=None):
 name=script.removeprefix('unified_').removesuffix('_20261004.py');log=R/f'work/unified_final_{name}.log';assert not log.exists(),f'Preserving existing log {log}'
 with log.open('w') as f:subprocess.run([str(python),str(R/'work'/script)]+list(args),cwd=R,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
 print('Completed',script,flush=True)
assert not (U/'final_test').exists() and not (U/'frozen_unified').exists()
while not (U/'paired_preview_complete.json').exists():
 path=Path('/proc/399524/cmdline');assert path.exists() and b'unified_run_pair_20261004.py' in path.read_bytes(),'Paired driver stopped without completion; inspect its log, never restart blindly'
 time.sleep(30)
expected=['joint_v1_validation','joint_v2_plain_validation','joint_v2_preview_validation','joint_v3_pose_common_step1000_validation','joint_v3_pose_common_validation']
expected += [f'joint_v4_tracking_step{i}_validation' for i in [1000,2000,3000,4000]]+['joint_v4_tracking_validation']
for variant in ['short','long']:expected += [f'joint_v5_root_{variant}_step{i}_validation' for i in [1000,3000]]+[f'joint_v5_root_{variant}_validation']
assert all((U/'evaluation'/n/'audit.json').exists() for n in expected)
run('unified_freeze_20261004.py',['--candidates']+expected)
selected=read(U/'frozen_unified/protocol.json');checkpoint=selected['checkpoint'];flags=['--preview'] if selected['preview'] else []
run('unified_generate_20261004.py',['--split','final_test'],GEN)
run('unified_retarget_20261004.py',['--split','final_test'])
rows=read(U/'final_test/manifest.json');assert len(rows)==880 and len({(x['source'],x['command_index']) for x in rows})==880
final_seeds={x['seed'] for x in rows};expected_seeds={94041000+p*100+s for p in range(4) for s in range(4)};assert final_seeds==expected_seeds
for split in ['training_extra','development_validation']:
 assert final_seeds.isdisjoint({x['seed'] for x in read(U/split/'manifest.json')})
for row in rows:
 assert sha(Path(row['generator']))==row['generator_sha256']
assert len(read(U/'final_test/reference_manifest.json'))==880
(U/'final_test/cohort_audit.json').write_text(json.dumps(dict(requests=880,unique_source_command_pairs=880,seed_families_disjoint=True,seeds=sorted(final_seeds),manifest_sha256=sha(U/'final_test/manifest.json'),reference_manifest_sha256=sha(U/'final_test/reference_manifest.json'),selected_checkpoint_sha256=selected['checkpoint_sha256'],scope='Fresh noise seeds within cached templates and trained command ranges; no unseen-text claim.'),indent=2))
run('unified_challenges_20261004.py',['--split','final_test'])
def unified():
 run('unified_evaluate_20261004.py',['--checkpoint',checkpoint,'--split','final_test','--name','unified_final']+flags,GPU)
 run('unified_assess_20261004.py',['--name','unified_final'])
def baseline():
 run('unified_baseline_20261004.py',['--split','final_test','--name','routed_baseline_final'])
 run('unified_assess_baseline_20261004.py',['--name','routed_baseline_final'])
def challenges():
 run('unified_evaluate_challenges_20261004.py',['--checkpoint',checkpoint,'--split','final_test','--name','unified_final_challenges']+flags,GPU)
 run('unified_assess_challenges_20261004.py',['--name','unified_final_challenges'])
with concurrent.futures.ThreadPoolExecutor(3) as pool:
 futures=[pool.submit(f) for f in [unified,baseline,challenges]]
 for future in futures:future.result()
run('unified_paired_statistics_20261004.py',['--model','unified_final','--baseline','routed_baseline_final','--output','statistics/final_paired_comparison.json'])
env=dict(os.environ,PYTHONPATH=str(R/'outputs_amass/g1_command_diagnosis_20261002_2346/plot_deps'))
run('unified_plot_comparison_20261004.py',['--names','unified_final','--labels','Unified-single-policy','--baseline','routed_baseline_final','--output','final_command_responses'],env=env)
run('unified_export_actor_20261004.py',['--checkpoint',checkpoint,'--output',str(U/'frozen_unified/actor.pt')],GPU)
(U/'final_pipeline_complete.json').write_text(json.dumps(dict(completed_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checkpoint_sha256=sha(Path(checkpoint)),selection=selected['selected'],required_next='Visual inspection, frozen entrypoint smoke, deliverable report/bundle and code push remain.'),indent=2))
print('Final simulations/audits/plot/export complete; inspect and package before declaring goal complete',flush=True)
