import sys,json,argparse,os
os.environ['OPENBLAS_NUM_THREADS']='1'
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/unified_generator_20261007';sys.path[:0]=[str(R/'work/universal_tracker_20261006'),str(R/'work')]
import unified_retarget_20261004 as rt
from fk_conversion import convert

def native(item):
 row,out=item;path=Path(out)/(Path(row['path']).stem+'_motion.npz');np.savez_compressed(path,fps=50.,**convert(np.load(row['reference_path'])['reference_qpos']));return dict(row,motion_path=str(path),split=row['split'])
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--workers',type=int,default=4);a=p.parse_args();folder=D/'generation'/a.name;rows=json.loads((folder/'manifest.json').read_text());split=json.loads((folder/'protocol.json').read_text())['split'];rows=[dict(r,split=split) for r in rows];refs=folder/'references';refs.mkdir(exist_ok=False);retargeted=[]
 with ProcessPoolExecutor(a.workers,initializer=rt.init) as pool:
  for r in pool.map(rt.work,[(r,str(refs)) for r in rows]):retargeted.append(r);print(len(retargeted),'retargeted',flush=True)
 (folder/'reference_manifest.json').write_text(json.dumps(retargeted,indent=2));native_dir=folder/'native_motion';native_dir.mkdir();converted=[]
 with ProcessPoolExecutor(a.workers) as pool:
  for r in pool.map(native,[(r,str(native_dir)) for r in retargeted]):converted.append(r)
 (folder/'native_manifest.json').write_text(json.dumps(converted,indent=2));print('finished',len(converted),flush=True)
