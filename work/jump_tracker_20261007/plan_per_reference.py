"""Inference-time, per-reference MuJoCo shooting around a frozen shared policy.
Not zero-shot neural tracking. Uses simulated trial rollouts of the given reference,
with a fixed algorithm/budget and no changes to robot limits or command/reference.
"""
import os
os.environ['OMP_NUM_THREADS']='1'
import sys,json,hashlib,argparse,time,shutil
from pathlib import Path
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import fast_planner as e
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/jump_tracker_20261007'
INITIAL=None

def initialize(initial,out):
 global INITIAL
 INITIAL=np.asarray(initial);e.initialize(str(D/'backup/actor.pt'),str(D/'backup/policy.pt'),out,'standing',False)

def optimize(item):
 index,row=item;started=time.monotonic();q=np.load(row['reference_path'])['reference_qpos'];target=e.tr.measure(e.tr.get_positions(e.G['source'],q),'jump',e.tr.robot_height(e.G['source']))['quantity'];r=dict(row,command=target);rng=np.random.default_rng(710712+index);mean=INITIAL.copy();std=np.r_[np.full(9,.12),.04,.15,.2,np.full(3,.15)];lo=np.r_[np.full(9,-.8),-.12,-.5,-2.,np.full(3,-.8)];hi=-lo;best=None;history=[];calls=0
 for generation in range(12):
  population=np.clip(mean+rng.normal(size=(24,15))*std,lo,hi);population[0]=INITIAL;population[1]=mean if best is None else best['parameters'];population[2]=np.zeros(15);results=[]
  for parameters in population:
   outcome=e.execute(dict(r,pulse_parameters=parameters.tolist()));calls+=1;results.append(dict(parameters=parameters.tolist(),loss=outcome.get('loss',300.)+.02*np.mean(parameters[:9]**2),outcome=outcome))
  results.sort(key=lambda x:x['loss']);best=results[0] if best is None or results[0]['loss']<best['loss'] else best;elite=np.array([x['parameters'] for x in results[:4]]);mean=.2*mean+.8*elite.mean(0);std=np.maximum(.2*std+.8*elite.std(0),np.r_[np.full(9,.02),.005,.025,.05,np.full(3,.025)]);history.append(dict(generation=generation,loss=best['loss'],outcome=best['outcome']))
  if generation>=2 and best['loss']<.25:break
 return dict(row,pulse_parameters=best['parameters'],planner_target_from_reference=target,planner_trials=calls,planner_wall_s=time.monotonic()-started,planner_optimization=best['outcome'],planner_history=history,planner_seed=710712+index)

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--manifest',required=True);p.add_argument('--name',required=True);p.add_argument('--initial',required=True);a=p.parse_args();out=D/'planning'/a.name;out.mkdir(parents=True,exist_ok=False);rows=json.loads(Path(a.manifest).read_text());initial=json.loads(Path(a.initial).read_text())['parameters'];assert len(initial)==15
 source=out/'source_snapshot';source.mkdir()
 for n in ['plan_per_reference.py','fast_planner.py','pulse_residual.py']:shutil.copy2(Path(__file__).parent/n,source/n)
 (out/'protocol.json').write_text(json.dumps(dict(scope=__doc__,manifest=a.manifest,manifest_sha256=hashlib.sha256(Path(a.manifest).read_bytes()).hexdigest(),initial=a.initial,initial_sha256=hashlib.sha256(Path(a.initial).read_bytes()).hexdigest(),max_generations=12,population=24,early_stop='After>=3generations, objective<0.25',target='Height derived from G1 reference geometry; requested command used only by later frozen evaluation',workers=4,base_sha256=hashlib.sha256((D/'backup/actor.pt').read_bytes()).hexdigest()),indent=2));results=[]
 with ProcessPoolExecutor(4,initializer=initialize,initargs=(initial,str(out))) as pool:
  for r in pool.map(optimize,enumerate(rows)):
   results.append(r);(out/'manifest.json').write_text(json.dumps(results,indent=2));print(len(results),r['command'],r['planner_optimization'],r['planner_trials'],flush=True)
 (out/'complete.json').write_text(json.dumps(dict(planned=len(rows),processed=len(results),total_trials=sum(r['planner_trials'] for r in results),summed_worker_s=sum(r['planner_wall_s'] for r in results)),indent=2))
