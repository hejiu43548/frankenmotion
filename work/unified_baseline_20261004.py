"""Frozen routed V3 control on the same references as unified development validation."""
import os,sys,json,hashlib,subprocess,argparse
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
from unified_evaluation_lock_20261004 import evaluation_slot
R=Path('/home/pku/frankenmotion');N=R/'outputs_amass/franken_improve_20261003';U=N.parent/'franken_unified_20261004';sys.path.insert(0,str(R/'work'))
import confirmation_eval_20261003 as ce
OUT=U/'evaluation/routed_baseline_validation';SPLIT=U/'development_validation'
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def init(out,split):
 global OUT,SPLIT
 OUT=Path(out);SPLIT=Path(split);ce.init()
def run(row):
 tr=ce.g.tr;stem=Path(row['path']).stem;ref=np.load(SPLIT/'references'/(stem+'_uniform.npz'))['reference_qpos'];arrays,fall=tr.prior_run.rollout(ce.g.M,ce.g.P,*tr.upsample(ref[:,7:],ref[:,3:7],ref[:,:3]));ts=(np.arange(len(arrays['qpos']))+1)*.02;want=1+np.arange(len(ref))*.05;actual=None
 if fall is None and ts[-1]>=want[-1]-1e-7:
  pos=tr.get_positions(ce.g.M,arrays['qpos']);pos=np.stack([np.interp(want,ts,x) for x in pos.reshape(len(pos),-1).T],1).reshape(len(ref),24,3);actual=tr.measure(pos,row['task'],tr.robot_height(ce.g.M))
 np.savez_compressed(OUT/'sonic'/(stem+'_uniform.npz'),reference_qpos=ref,time_s=ts,**arrays);r=dict(row,actual=actual,fall_time=fall,controller='sonic',g1=tr.measure(tr.get_positions(ce.g.M,ref),row['task'],tr.robot_height(ce.g.M)));(OUT/'sonic'/(stem+'.json')).write_text(json.dumps(r,default=lambda x:x.item()));return r
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--split',choices=['development_validation','final_test'],default='development_validation');p.add_argument('--name',default='routed_baseline_validation');a=p.parse_args();OUT=U/'evaluation'/a.name;SPLIT=U/a.split
 if a.split=='final_test':assert (U/'frozen_unified/protocol.json').exists()
 OUT.mkdir(parents=True,exist_ok=False);(OUT/'sonic').mkdir();rows=json.loads((SPLIT/'manifest.json').read_text());cp=json.loads((N/'frozen_controllers_v1/protocol.json').read_text());route=cp['task_route']
 for v in cp['weights'].values():assert sha(v['frozen'])==v['sha256']
 proto=dict(split=a.split,scope='Frozen routed V3 baseline; explicitly NOT a unified policy',manifest_sha256=sha(SPLIT/'manifest.json'),task_route=route,controller_protocol_sha256=sha(N/'frozen_controllers_v1/protocol.json'));(OUT/'protocol.json').write_text(json.dumps(proto,indent=2));ss=[r for r in rows if route[r['task']]=='sonic'];bb=[r for r in rows if route[r['task']]=='beyondmimic'];assert len(rows)==(110 if a.split=='development_validation' else 880) and len(ss)*3==len(bb)*8;result=[]
 with ProcessPoolExecutor(4,initializer=init,initargs=(str(OUT),str(SPLIT))) as pool:
  for r in pool.map(run,ss):result.append(r);print(len(result),'SONIC baseline complete',flush=True)
 (OUT/'sonic/results.json').write_text(json.dumps(result,indent=2,default=lambda x:x.item()));(OUT/'bm_manifest.json').write_text(json.dumps(bb,indent=2));env=dict(os.environ,BM_CPU_FK='1',BM_ENTRY='standing',BM_TERMINATION='physical',BM_OUT=str(OUT/'bm'),BM_RESULTS=str(OUT/'bm_manifest.json'),BM_REFERENCE_DIR=str(SPLIT/'references'),BM_WEIGHT_MAP=str(N/'frozen_controllers_v1/weight_map.json'),BM_QUIET_METRICS='1');env.pop('BM_TASK',None);env.pop('BM_CHECKPOINT',None)
 with evaluation_slot(), (OUT/'bm.log').open('w') as log:subprocess.run([str(R/'work/mjlab_stable_env/bin/python'),str(R/'work/mjlab_probe_capacity_20261003.py')],env=env,cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
 assert 'overflow' not in (OUT/'bm.log').read_text().lower();print('Baseline simulations complete; raw audit required',flush=True)
