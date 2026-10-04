"""One actor and continuous physics for every complete challenge reference."""
import os,json,hashlib,argparse,subprocess
from pathlib import Path
from unified_evaluation_lock_20261004 import evaluation_slot
from unified_preview_20261004 import checkpoint_offsets
R=Path('/home/pku/frankenmotion');U=R/'outputs_amass/franken_unified_20261004'
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--preview',action='store_true');p.add_argument('--split',choices=['development_validation','final_test'],default='development_validation');p.add_argument('--name',required=True);a=p.parse_args();ck=Path(a.checkpoint);sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
if a.split=='final_test':assert json.loads((U/'frozen_unified/protocol.json').read_text())['checkpoint_sha256']==sha(ck)
source=U/a.split/'challenges';out=U/'evaluation'/a.name;out.mkdir(parents=True,exist_ok=False);offsets=checkpoint_offsets(ck);assert bool(offsets)==a.preview
if a.split=='final_test':
 selected=json.loads((U/'frozen_unified/protocol.json').read_text());assert list(offsets)==selected.get('preview_offsets',[5,10,20] if selected['preview'] else [])
proto=dict(preview_offsets=list(offsets),checkpoint=str(ck),sha256=sha(ck),preview=a.preview,source=str(source),split=a.split,manifest_sha256=sha(source/'manifest.json'),single_policy=True,task_routing=False,standing_entries_per_reference=1)
(out/'protocol.json').write_text(json.dumps(proto,indent=2));env=dict(os.environ,UNIFIED_PREVIEW='1' if a.preview else '0',BM_CPU_FK='1',BM_ENTRY='standing',BM_TERMINATION='physical',BM_OUT=str(out),BM_RESULTS=str(source/'manifest.json'),BM_REFERENCE_DIR=str(source/'references'),BM_CHECKPOINT=str(ck),BM_QUIET_METRICS='1');env.pop('BM_WEIGHT_MAP',None);env.pop('BM_TASK',None)
with evaluation_slot(), (out/'worker.log').open('w') as log:subprocess.run([str(R/'work/mjlab_stable_env/bin/python'),str(R/'work/unified_challenge_probe_20261004.py')],env=env,cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
assert 'overflow' not in (out/'worker.log').read_text().lower() and sha(ck)==proto['sha256'];print('Continuous challenge simulations finished',out,flush=True)
