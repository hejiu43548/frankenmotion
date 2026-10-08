"""Predeclared second seed of the selected expanded continuation, no reselection.
Warm-start v4 checkpoint is shared; this is not an independent full-pipeline seed.
"""
import os,json,time,hashlib,subprocess,datetime
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/universal_tracker_20261006';W=R/'work/universal_tracker_20261006';py=R/'work/mjlab_stable_env/bin/python';seed=6107;name='replication_v5_seed6107';cutoff=datetime.datetime(2026,10,6,2,5,tzinfo=datetime.timezone.utc).timestamp()
def status(stage,**kw):(D/'replication_status.json').write_text(json.dumps(dict(stage=stage,updated_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),**kw),indent=2))
def wait_for(paths):
 while not all(p.exists() for p in paths):
  if time.time()>cutoff:raise TimeoutError('Second-seed study time limit')
  time.sleep(10)
def run(label,command):
 logpath=D/('replication_'+label+'.log')
 with logpath.open('w') as log:
  proc=subprocess.Popen(list(map(str,command)),cwd=R,env=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1'),stdout=log,stderr=subprocess.STDOUT)
  while proc.poll() is None:
   if time.time()>cutoff:
    proc.terminate()
    try:proc.wait(timeout=30)
    except subprocess.TimeoutExpired:proc.kill();proc.wait()
    raise TimeoutError(('Owned replication process stopped at declared deadline',label))
   time.sleep(5)
 if proc.returncode:raise RuntimeError((label,proc.returncode,str(logpath)))
try:
 status('waiting_for_freeze');wait_for([D/'frozen_unified/protocol.json',D/'training/joint_v5_expanded/complete.json']);f=json.loads((D/'frozen_unified/protocol.json').read_text());assert f['selected'].startswith('v5_'),f['selected'];source=Path(f['source_checkpoint']);step=int(source.stem.split('_')[-1]);base=json.loads((D/'training/joint_v5_expanded/protocol.json').read_text());args=dict(base['arguments'],name=name,steps=step+1,seed=seed);command=[str(py),str(W/'train_broad.py')]
 for key,value in args.items():
  flag='--'+key.replace('_','-')
  if isinstance(value,bool):
   if value:command.append(flag)
  elif value is not None:command.extend([flag,str(value)])
 protocol=dict(scope=__doc__,selected_main=f['selected'],main_checkpoint_sha256=f['checkpoint_sha256'],target_iteration=step,new_seed=seed,source_arguments=base['arguments'],replica_arguments=args,command=command,selection='Target iteration comes from first-seed global development selection. Exactly one continuation seed; no checkpoint selection within replica and no deployment switch. Final tests are reused only for reporting after both policies are fixed.',time_limit_utc='2026-10-06T02:05:00Z');(D/'replication_protocol.json').write_text(json.dumps(protocol,indent=2));status('training',target_iteration=step,seed=seed);run('training',command)
 ck=D/'training'/name/f'model_{step}.pt';ready=json.loads(ck.with_suffix('.pt.ready.json').read_text());assert hashlib.sha256(ck.read_bytes()).hexdigest()==ready['checkpoint_sha256'];status('waiting_for_fresh_references',checkpoint=str(ck),checkpoint_sha256=ready['checkpoint_sha256']);wait_for([D/'fresh_final/native_manifest.json',D/'natural_test_motion/manifest.json',D/'fresh_table/sequences/manifest.json']);status('evaluation',checkpoint=str(ck),checkpoint_sha256=ready['checkpoint_sha256'])
 for label,manifest,tag in [('eleven',D/'fresh_final/native_manifest.json','replica_seed6107_eleven'),('natural',D/'natural_test_motion/manifest.json','replica_seed6107_natural')]:
  run(label,[py,W/'evaluate_cpu_general.py','--checkpoint',ck,'--manifest',manifest,'--name',tag,'--workers',4]);run(label+'_audit',[py,W/('assess_native_eleven.py' if label=='eleven' else 'assess_general_fidelity.py'),*(['--name',tag] if label=='eleven' else ['--run',D/'general_evaluation'/tag])])
 run('table',[py,W/'evaluate_table_alternative.py','--checkpoint',ck,'--scenes-folder',D/'fresh_table/sequences','--name','replica_seed6107','--workers',2,'--scope','Predeclared second continuation seed; no reselection or deployment switch. Same frozen fresh references.'])
 status('complete',checkpoint=str(ck),checkpoint_sha256=ready['checkpoint_sha256'],target_iteration=step,seed=seed);print('Second-seed continuation and fixed tests complete',flush=True)
except Exception as exc:status('failed_or_incomplete',error=repr(exc));raise
