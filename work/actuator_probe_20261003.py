"""Ablate verified ankle/waist effort and reflected inertia mismatch."""
import sys,json
from pathlib import Path
sys.path.insert(0,'/home/pku/frankenmotion/work')
import v11_20261003 as v
import numpy as np
from concurrent.futures import ProcessPoolExecutor
BASE=v.BASE;OUT=v.OUT;P=None;M=None

def init():
 global P,M
 v.tr.rt.EFF[[4,5,10,11,13,14]]=50.
 v.tr.rt.ARM[[4,5,10,11,13,14]]=2*.003609725
 M=v.tr.rt.load_model();P=[v.tr.rt.Policy(),v.Policy(0)]
def run(row):
 folder=OUT/'actuator_probe';folder.mkdir(exist_ok=True,parents=True);name=Path(row['path']).stem;dest=folder/(name+'.json')
 if dest.exists():return json.loads(dest.read_text())
 z=np.load(BASE/'simulation/task_adapter'/(name+'_reference.npz'));q,quat,root=z['q'],z['quat'],z['root'];r=dict(row,modes={});up=v.tr.upsample(q,quat,root)
 for label,policy in zip(['default_fixed','v11_fixed'],P):
  arrays,fall=v.tr.prior_run.rollout(M,policy,*up);ts=(np.arange(len(arrays['qpos']))+1)*.02;want=1+np.arange(len(q))*.05;metrics=None
  if fall is None and ts[-1]>=want[-1]-1e-7:
   ps=v.tr.get_positions(M,arrays['qpos']);p=np.stack([np.interp(want,ts,x) for x in ps.reshape(len(ps),-1).T],1).reshape(len(q),24,3);metrics=v.tr.measure(p,row['task'],v.tr.robot_height(M))
  r['modes'][label]=dict(metrics=metrics,fall=fall);np.savez_compressed(folder/(name+'_'+label+'.npz'),**arrays,time_s=ts)
 dest.write_text(json.dumps(r,default=lambda x:x.item()));return r
if __name__=='__main__':
 rows=json.loads((BASE/'generated/task_adapter_manifest.json').read_text());rows=[r for r in rows if r['source'].endswith('_p0_s0')];results=[]
 with ProcessPoolExecutor(4,initializer=init) as pool:
  for r in pool.map(run,rows):results.append(r);print(len(results),r['task'],flush=True)
 (OUT/'actuator_probe_results.json').write_text(json.dumps(results,indent=2,default=lambda x:x.item()))
