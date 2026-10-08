import os,json,hashlib,datetime,subprocess
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/jump_tracker_20261007';W=R/'work/jump_tracker_20261007'
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
plan=json.load(open(D/'plan.json'));freeze=json.load(open(D/'candidate_freeze.json'));generation=json.load(open(D/'fresh_final/protocol.json'));assert sha(D/'candidate_freeze.json')==generation['selection_sha256'];assert sha(Path(freeze['checkpoint']))==freeze['checkpoint_sha256']
backup={}
for name in ['actor.pt','policy.pt','protocol.json']:
 expected=plan['backup'][name];assert sha(D/'backup'/name)==expected;assert sha(R/'outputs_amass/universal_tracker_20261006/frozen_unified'/name)==expected;backup[name]=expected
base=json.load(open(D/'general_evaluation/final_baseline/audited_results.json'));new=json.load(open(D/'general_evaluation/final_candidate/audited_results.json'));key=lambda r:(r['source'],r['seed'],round(r['command'],6));old={key(r):r for r in base};assert len(old)==len(new)==60
for r in new:
 b=old[key(r)];assert b['reference_path']==r['reference_path'] and b['motion_path']==r['motion_path'];assert sha(Path(r['path']))==r['human_sha256'];assert sha(Path(r['generator']))==r['generator_sha256']
teacher=json.load(open(D/'teacher_train_manifest.json'));extra=json.load(open(D/'expanded_training/manifest.json'));oldseeds={r.get('seed') for r in teacher+extra};assert not {r['seed'] for r in new}&oldseeds
states={p.name:json.load(open(p)) for p in D.glob('*process.json')};pending=[n for n,r in states.items() if 'returncode' not in r];assert not pending,pending
processes=[]
for line in subprocess.check_output(['ps','-eo','pid,args'],text=True).splitlines()[1:]:
 parts=line.strip().split(None,1)
 if len(parts)!=2:continue
 pid,args=parts
 if int(pid)==os.getpid():continue
 executable=Path(args.split()[0]).name
 if 'python' in executable and 'work/jump_tracker_20261007/' in args:processes.append(dict(pid=int(pid),args=args))
assert not processes,processes
now=datetime.datetime.now(datetime.timezone.utc);start=datetime.datetime.fromisoformat(plan['start_utc'].replace('Z','+00:00'));end=datetime.datetime.fromisoformat(plan['hard_end_utc'].replace('Z','+00:00'));assert now<end
result=dict(finished_utc=now.isoformat(),elapsed_minutes=(now-start).total_seconds()/60,hard_end_utc=plan['hard_end_utc'],within_three_hours=True,active_task_python_processes=processes,all_budgeted_processes_finished=True,original_frozen_and_backup_hashes_verified=backup,candidate_sha256=freeze['checkpoint_sha256'],candidate_unchanged_since_freeze=True,fresh_noise_disjoint_from_teacher=True,identical_references_between_trackers=True,requests=60,pushed=False,baseline_replaced=False,scope='Bounded improvement attempt completed, not a claim of universal jump reliability or hardware readiness.',source_hashes={p.name:sha(p) for p in W.glob('*.py')},nonzero_runs_preserved={n:r['returncode'] for n,r in states.items() if r['returncode']})
(D/'finalization.json').write_text(json.dumps(result,indent=2));print(json.dumps({k:v for k,v in result.items() if k!='source_hashes'},indent=2))
