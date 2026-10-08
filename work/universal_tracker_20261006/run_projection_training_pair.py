"""Post-freeze exploratory matched training pair; never use final-test data or replace release."""
import os,json,time,subprocess,datetime,signal
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/universal_tracker_20261006';W=R/'work/universal_tracker_20261006';py=R/'work/mjlab_stable_env/bin/python';cutoff=datetime.datetime(2026,10,6,2,5,tzinfo=datetime.timezone.utc).timestamp();out=D/'rollout_projection_study';out.mkdir(exist_ok=True)
def status(stage,**kw):(out/'status.json').write_text(json.dumps(dict(stage=stage,utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw),indent=2))
def run(label,command):
 with (out/(label+'.log')).open('w') as log:
  p=subprocess.Popen(list(map(str,command)),cwd=R,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
  while p.poll() is None:
   if time.time()>cutoff:
    os.killpg(p.pid,signal.SIGTERM)
    try:p.wait(timeout=10)
    except subprocess.TimeoutExpired:os.killpg(p.pid,signal.SIGKILL);p.wait()
    raise TimeoutError(label)
   time.sleep(3)
  if p.returncode:raise RuntimeError((label,p.returncode))
def evaluate(ck,tag):
 for kind,manifest in [('eleven',D/'native_eleven_manifest.json'),('natural',D/'natural_extended_corrected_motion/manifest.json')]:
  name='projection_'+tag+'_'+kind;args=[py,W/'evaluate_cpu_general.py','--checkpoint',ck,'--manifest',manifest,'--name',name,'--workers',3]
  if kind=='natural':args+=['--split','val']
  run(name,args);run(name+'_audit',[py,W/('assess_native_eleven.py' if kind=='eleven' else 'assess_general_fidelity.py'),*(['--name',name] if kind=='eleven' else ['--run',D/'general_evaluation'/name])])
 run('projection_'+tag+'_table',[py,W/'evaluate_table.py','--checkpoint',ck,'--name','projection_'+tag,'--workers',2])
 return dict(eleven=json.loads((D/f'general_evaluation/projection_{tag}_eleven/eleven_audit.json').read_text())['aggregate'],natural=json.loads((D/f'general_evaluation/projection_{tag}_natural/fidelity_summary.json').read_text()),table=json.loads((D/f'table_evaluation/projection_{tag}/summary.json').read_text()))
try:
 status('waiting_for_replica_training_and_projected_corpus')
 while not (D/'training/replication_v5_seed6107/complete.json').exists() or not (out/'projection_audit.json').exists():
  if time.time()>cutoff-1800:raise TimeoutError('Insufficient time for matched pair; do not substitute unequal budget results')
  time.sleep(10)
 base=json.loads((D/'training/joint_v5_expanded/protocol.json').read_text())['arguments'];plan=dict(scope=__doc__,declared_updates=2000,seed=6108,initial=str(D/'frozen_unified/policy.pt'),dataset_pair=['joint_corpus_expanded_corrected','joint_corpus_rollout_projected'],evaluation='Development native110 + natural57 + table6 only. Fixed last checkpoint for both, no per-category selection.',time_limit_utc='2026-10-06T02:05:00Z');(out/'training_protocol.json').write_text(json.dumps(plan,indent=2));futures={}
 with ThreadPoolExecutor(2) as pool:
  futures['initial']=pool.submit(evaluate,D/'frozen_unified/policy.pt','initial')
  for tag,dataset in [('raw','joint_corpus_expanded_corrected'),('projected','joint_corpus_rollout_projected')]:
   name='projection_pair_'+tag;args=dict(base,name=name,dataset=dataset,initial=plan['initial'],steps=2000,seed=6108);command=[py,W/'train_broad.py']
   for k,v in args.items():
    flag='--'+k.replace('_','-')
    if isinstance(v,bool):
     if v:command.append(flag)
    elif v is not None:command.extend([flag,v])
   status('training_'+tag);run('train_'+tag,command);ck=D/'training'/name/'model_1999.pt';assert ck.with_suffix('.pt.ready.json').exists();futures[tag]=pool.submit(evaluate,ck,tag)
  status('evaluating');results={tag:f.result() for tag,f in futures.items()}
 (out/'results.json').write_text(json.dumps(dict(protocol=plan,projection=json.loads((out/'projection_audit.json').read_text()),results=results),indent=2));status('complete');print('Matched development study complete',flush=True)
except Exception as exc:status('incomplete_or_failed',error=repr(exc));raise
