"""Diagnostic routed baseline with one native BM action/physics interface; not a unified policy."""
import os,sys,json,hashlib,argparse,subprocess
from pathlib import Path
R=Path('/home/pku/frankenmotion');U=R/'outputs_amass/franken_unified_20261004'
p=argparse.ArgumentParser();p.add_argument('--split',default='development_validation');p.add_argument('--name',required=True);a=p.parse_args();wm=R/'outputs_amass/franken_improve_20261003/frozen_controllers_v1/weight_map.json';weights=json.loads(wm.read_text());hashes={v:hashlib.sha256(Path(v).read_bytes()).hexdigest() for v in set(weights.values())};out=U/'evaluation'/a.name;out.mkdir(parents=True,exist_ok=False)
assert a.split!='final_test','Final evaluation requires a separately frozen selection protocol'
proto=dict(preview=False,weight_map=weights,weight_sha256=hashes,split=a.split,single_shared_checkpoint=False,task_routing=True,scope='BM native routed control',manifest_sha256=hashlib.sha256((U/a.split/'manifest.json').read_bytes()).hexdigest())
(out/'protocol.json').write_text(json.dumps(proto,indent=2));env=dict(os.environ,UNIFIED_PREVIEW='0',BM_CPU_FK='1',BM_ENTRY='standing',BM_TERMINATION='physical',BM_OUT=str(out),BM_RESULTS=str(U/a.split/'manifest.json'),BM_REFERENCE_DIR=str(U/a.split/'references'),BM_WEIGHT_MAP=str(wm),BM_QUIET_METRICS='1');env.pop('BM_CHECKPOINT',None);env.pop('BM_TASK',None)
with (out/'worker.log').open('w') as log:subprocess.run([str(R/'work/mjlab_stable_env/bin/python'),str(R/'work/mjlab_probe_capacity_20261003.py')],env=env,cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
assert 'overflow' not in (out/'worker.log').read_text().lower();assert all(hashlib.sha256(Path(k).read_bytes()).hexdigest()==v for k,v in hashes.items());print('Native routed diagnostic control complete',out,flush=True)
