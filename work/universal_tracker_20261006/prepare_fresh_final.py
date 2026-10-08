"""Frozen GMR and CPU FK for fresh test; never consult tracker outcomes."""
import sys,json,os,hashlib
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/universal_tracker_20261006';sys.path.insert(0,str(R/'work'))
import unified_retarget_20261004 as rt
import numpy as np
from fk_conversion import convert

def native(row):
 path=D/'fresh_final/native_motion'/(Path(row['path']).stem+'_motion.npz');np.savez_compressed(path,fps=50.,**convert(np.load(row['reference_path'])['reference_qpos']));return dict(row,motion_path=str(path),split='fresh_final')
if __name__=='__main__':
 folder=D/'fresh_final';generation=json.loads((folder/'protocol.json').read_text());assert generation['selection_sha256']==hashlib.sha256((D/'frozen_unified/protocol.json').read_bytes()).hexdigest();rows=json.loads((folder/'manifest.json').read_text());assert len(rows)==880;(folder/'references').mkdir(exist_ok=False);retargeted=[]
 with ProcessPoolExecutor(4,initializer=rt.init) as pool:
  for row in pool.map(rt.work,[(r,str(folder/'references')) for r in rows]):retargeted.append(row);print(len(retargeted),'retargeted',flush=True)
 (folder/'reference_manifest.json').write_text(json.dumps(retargeted,indent=2));(folder/'native_motion').mkdir();converted=[]
 with ProcessPoolExecutor(4) as pool:
  for row in pool.map(native,retargeted):converted.append(row);print(len(converted),'FK',flush=True)
 (folder/'native_manifest.json').write_text(json.dumps(converted,indent=2));subset=D/'fresh_gpu_subset';subset.mkdir();selected=[r for r in rows if r['source'].endswith(('_p0_s0','_p2_s2'))];assert len(selected)==110;(subset/'manifest.json').write_text(json.dumps(selected,indent=2));os.symlink(folder/'references',subset/'references',target_is_directory=True);(subset/'native_manifest.json').write_text(json.dumps([r for r in converted if r['source'].endswith(('_p0_s0','_p2_s2'))],indent=2))
 (folder/'prepared.json').write_text(json.dumps(dict(requests=len(converted),gpu_subset=len(selected),native_manifest_sha256=hashlib.sha256((folder/'native_manifest.json').read_bytes()).hexdigest(),source_manifest_sha256=hashlib.sha256((folder/'manifest.json').read_bytes()).hexdigest()),indent=2))
