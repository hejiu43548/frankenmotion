"""Frozen V3 retarget only, without evaluating or choosing a tracker."""
import sys,os,json,argparse,hashlib,importlib.util
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import numpy as np
R=Path('/home/pku/frankenmotion');N=R/'outputs_amass/franken_improve_20261003';U=N.parent/'franken_unified_20261004';sys.path.insert(0,str(R/'work'))
import gmr_probe_20261003 as g
F=None
def init():
 global F
 g.M=g.tr.rt.load_model();g.P=None;p=N/'frozen_retarget_v2';proto=json.loads((p/'protocol.json').read_text());script=p/'task_retarget_v2_20261003.py';assert hashlib.sha256(script.read_bytes()).hexdigest()==proto['script_sha256'];spec=importlib.util.spec_from_file_location('frozen_refiner',script);F=importlib.util.module_from_spec(spec);spec.loader.exec_module(F)
def work(item):
 row,out=item;z=np.load(row['path']);ref=g.convert(z,'uniform')
 if row['task'] in ['raise_hand','lean']:ref=F.refine(z,ref,row['task'],4)
 path=Path(out)/(Path(row['path']).stem+'_uniform.npz');np.savez_compressed(path,reference_qpos=ref);return dict(row,reference_path=str(path))
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--split',required=True);a=p.parse_args();folder=U/a.split;out=folder/'references';out.mkdir(exist_ok=False);rows=json.loads((folder/'manifest.json').read_text());result=[]
 with ProcessPoolExecutor(4,initializer=init) as pool:
  for r in pool.map(work,[(r,str(out)) for r in rows]):result.append(r);print(len(result),'retargeted',flush=True)
 (folder/'reference_manifest.json').write_text(json.dumps(result,indent=2))
