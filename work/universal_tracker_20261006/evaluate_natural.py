import sys
sys.path.insert(0,"/home/pku/frankenmotion/work")
"""Evaluate exactly one checkpoint on every task with the established physical protocol."""
import os,sys,json,hashlib,argparse,subprocess
from pathlib import Path
from unified_evaluation_lock_20261004 import evaluation_slot
from unified_preview_20261004 import checkpoint_offsets
R=Path('/home/pku/frankenmotion');U=R/'outputs_amass/universal_tracker_20261006'
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--preview',action='store_true');p.add_argument('--split',default='development_validation');p.add_argument('--name',required=True);a=p.parse_args();ck=Path(a.checkpoint);out=U/'evaluation'/a.name;out.mkdir(parents=True,exist_ok=False)
if a.split=='final_test':
 frozen=json.loads((U/'frozen_unified/protocol.json').read_text());assert hashlib.sha256(ck.read_bytes()).hexdigest()==frozen['checkpoint_sha256'] and a.preview==frozen['preview'] and not frozen['task_routing']
offsets=checkpoint_offsets(ck);assert bool(offsets)==a.preview
if a.split=='final_test':
 selected=json.loads((U/'frozen_unified/protocol.json').read_text());assert list(offsets)==selected.get('preview_offsets',[5,10,20] if selected['preview'] else [])
proto=dict(preview_offsets=list(offsets),preview=a.preview,checkpoint=str(ck),sha256=hashlib.sha256(ck.read_bytes()).hexdigest(),split=a.split,single_shared_checkpoint=True,task_routing=False,manifest_sha256=hashlib.sha256((U/a.split/'manifest.json').read_bytes()).hexdigest())
(out/'protocol.json').write_text(json.dumps(proto,indent=2));env=dict(os.environ,UNIFIED_PREVIEW='1' if a.preview else '0',BM_CPU_FK='1',BM_ENTRY='standing',BM_TERMINATION='physical',BM_OUT=str(out),BM_RESULTS=str(U/a.split/'manifest.json'),BM_REFERENCE_DIR=str(U/a.split/'references'),BM_CHECKPOINT=str(ck),BM_QUIET_METRICS='1');env.pop('BM_WEIGHT_MAP',None);env.pop('BM_TASK',None)
with evaluation_slot(), (out/'worker.log').open('w') as log:subprocess.run([str(R/'work/mjlab_stable_env/bin/python'),str(R/'work/universal_tracker_20261006/natural_probe.py')],env=env,cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
assert 'overflow' not in (out/'worker.log').read_text().lower();assert hashlib.sha256(ck.read_bytes()).hexdigest()==proto['sha256'];print('Single-checkpoint evaluation complete',out,flush=True)
