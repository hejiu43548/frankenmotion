"""Frozen integrated primary pipeline: SONIC640, GMR-only references for BM240."""
import sys,json,importlib.util,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
sys.path.insert(0,'/home/pku/frankenmotion/work')
import confirmation_eval_20261003 as ce
ROOT=ce.ROOT;OUT=ROOT/'integrated_v3_sonic';OUT.mkdir(exist_ok=True);protocol=json.loads((ROOT/'frozen_integrated_v3/protocol.json').read_text());ROUTE=protocol['task_route']
def init():
 ce.init();f=ROOT/'frozen_retarget_v2/task_retarget_v2_20261003.py';p=json.loads((f.parent/'protocol.json').read_text());assert hashlib.sha256(f.read_bytes()).hexdigest()==p['script_sha256'];spec=importlib.util.spec_from_file_location('refiner',f);r=importlib.util.module_from_spec(spec);spec.loader.exec_module(r);original=ce.g.convert
 def convert(z,variant):
  states=original(z,variant);task=str(z['task']);return r.refine(z,states,task,4) if task in ['raise_hand','lean'] else states
 ce.g.convert=convert

def run(row):
 if ROUTE[row['task']]=='sonic':
  r=ce.run((row,'uniform',str(OUT)));r['execution']='sonic';return r
 stem=Path(row['path']).stem;r=dict(row,method='uniform',execution='beyondmimic_pending')
 try:
  states=ce.g.convert(ce.g.np.load(row['path']),'uniform');r['g1']=ce.g.tr.measure(ce.g.tr.get_positions(ce.g.M,states),row['task'],ce.g.tr.robot_height(ce.g.M));ce.g.np.savez_compressed(OUT/(stem+'_uniform.npz'),reference_qpos=states)
 except Exception as e:r['error']=repr(e)
 (OUT/(stem+'_uniform.json')).write_text(json.dumps(r,default=lambda x:x.item()));return r
if __name__=='__main__':
 rows=json.loads((ROOT/'integrated_v3_manifest.json').read_text());assert len(rows)==880;results=[]
 with ProcessPoolExecutor(4,initializer=init) as pool:
  for r in pool.map(run,rows):
   results.append(r)
   if len(results)%40==0:print(len(results),'requests processed',flush=True)
 (OUT/'results.json').write_text(json.dumps(results,indent=2,default=lambda x:x.item()))
