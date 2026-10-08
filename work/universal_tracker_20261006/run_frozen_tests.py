"""Run declared tests after policy freeze. Never changes weights or selects policies."""
import os,json,hashlib,subprocess,datetime,argparse
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor,as_completed
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/universal_tracker_20261006';W=R/'work/universal_tracker_20261006';PY=R/'work/mjlab_stable_env/bin/python';GEN=R/'.conda/bin/python'
p=argparse.ArgumentParser();p.add_argument('--phase',choices=['prepare','cpu','gpu','robustness'],required=True);a=p.parse_args()
freeze=json.loads((D/'frozen_unified/protocol.json').read_text());candidate=Path(freeze['checkpoint']);assert hashlib.sha256(candidate.read_bytes()).hexdigest()==freeze['checkpoint_sha256'] and not freeze['task_routing']
controllers={'candidate':candidate,'broad':D/'backup/prior_unified_frozen/policy.pt','stable':D/'backup/stable_frozen/policy.pt'}
logs=D/'frozen_test_logs';logs.mkdir(exist_ok=True)
def run(tag,script,args=(),python=PY,env=None):
 command=[str(python),str(W/script),*map(str,args)];print(datetime.datetime.now(datetime.timezone.utc).isoformat(),tag,'start',flush=True)
 environment=dict(os.environ,OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1');environment.update(env or {})
 with (logs/(tag+'.log')).open('w') as f:r=subprocess.run(command,cwd=R,env=environment,stdout=f,stderr=subprocess.STDOUT)
 (logs/(tag+'.status.json')).write_text(json.dumps(dict(command=command,returncode=r.returncode,numerical_thread_defaults={k:environment[k] for k in ['OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS']},note='Scripts may explicitly set Torch thread count, e.g. generation4 or inference1.',finished_utc=datetime.datetime.now(datetime.timezone.utc).isoformat()),indent=2))
 if r.returncode:raise RuntimeError((tag,r.returncode))
 print(tag,'complete',flush=True)
def parallel(function,items,workers):
 errors=[]
 with ThreadPoolExecutor(workers) as pool:
  futures={pool.submit(function,item):item for item in items}
  for f in as_completed(futures):
   try:f.result()
   except Exception as e:errors.append((str(futures[f]),repr(e)));print('FAILED',errors[-1],flush=True)
 if errors:raise RuntimeError(errors)
if a.phase=='prepare':
 def prepare(tag):
  if tag=='eleven':
   run('generate_fresh_final','generate_fresh_final.py',python=GEN);run('prepare_fresh_final','prepare_fresh_final.py')
  elif tag=='table':
   run('generate_fresh_table','generate_fresh_table.py',python=GEN);run('prepare_fresh_table','prepare_fresh_table.py')
  else:
   run('prepare_natural_test','prepare_natural_test.py');env=dict(NATURAL_SET='natural_test',NATURAL_MOTION_SET='natural_test_motion')
   run('retarget_natural_test','retarget_natural_candidates.py',env=env);run('fk_natural_test','prepare_natural_motion.py',env=env)
 parallel(prepare,['eleven','table','natural'],2)
elif a.phase=='cpu':
 def evaluate(tag):
  if tag=='sonic':run('fresh_sonic','evaluate_sonic_native.py',['--manifest',D/'fresh_final/native_manifest.json','--name','fresh_final','--workers',4]);return
  checkpoint=controllers[tag];name='fresh_'+tag
  run(name,'evaluate_cpu_general.py',['--checkpoint',checkpoint,'--manifest',D/'fresh_final/native_manifest.json','--name',name,'--workers',4])
  run(name+'_audit','assess_native_eleven.py',['--name',name])
  run(name+'_natural','evaluate_cpu_general.py',['--checkpoint',checkpoint,'--manifest',D/'natural_test_motion/manifest.json','--name',name+'_natural','--workers',3])
  run(name+'_natural_audit','assess_general_fidelity.py',['--run',D/'general_evaluation'/(name+'_natural')])
  run(name+'_table','evaluate_table_alternative.py',['--checkpoint',checkpoint,'--scenes-folder',D/'fresh_table/sequences','--name',name,'--workers',2,'--scope','Frozen fresh paired layouts and generator noise, all12 retained; no checkpoint selection from this test.'])
 parallel(evaluate,['candidate','broad','stable','sonic'],4)
 for baseline in ['broad','stable']:
  run('fresh_statistics_'+baseline,'paired_statistics.py',['--candidate',D/'general_evaluation/fresh_candidate/audited_results.json','--baseline',D/('general_evaluation/fresh_'+baseline+'/audited_results.json'),'--output',D/('fresh_statistics_'+baseline+'.json')])
elif a.phase=='gpu':
 for tag in ['candidate','broad']:
  run('fresh_gpu_'+tag,'evaluate.py',['--checkpoint',controllers[tag],'--preview','--split','fresh_gpu_subset','--name','fresh_gpu_'+tag])
  run('fresh_gpu_'+tag+'_audit','assess.py',['--name','fresh_gpu_'+tag],python=R/'work/g1_sim_env/bin/python')
else:
 def robust(item):
  tag,profile=item;name='fresh_robust_'+tag+'_'+profile
  run(name,'evaluate_cpu_robustness.py',['--checkpoint',controllers[tag],'--manifest',D/'fresh_gpu_subset/native_manifest.json','--name',name,'--workers',4,'--profile',profile])
  run(name+'_audit','assess_native_eleven.py',['--name',name])
 parallel(robust,[(tag,profile) for tag in ['candidate','broad'] for profile in ['nominal','friction_0p6','mass_1p1','delay_20ms','lateral_push_40N']],2)
