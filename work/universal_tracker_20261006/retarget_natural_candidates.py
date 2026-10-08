"""Uniform GMR only, no task-specific refiner, no controller selection."""
import sys,json,hashlib,time,os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/universal_tracker_20261006'
sys.path.insert(0,str(R/'work'))
import gmr_probe_20261003 as g
import numpy as np

def init():
    g.M=g.tr.rt.load_model();g.P=None

def convert(row):
    start=time.monotonic()
    try:
        z=np.load(row['path']);q=g.convert(z,'uniform')
        assert np.isfinite(q).all()
        dest=D/os.environ.get('NATURAL_SET','natural_candidates')/'references'/(Path(row['path']).stem+'_uniform.npz')
        np.savez_compressed(dest,reference_qpos=q)
        return dict(row,reference_path=str(dest),reference_sha256=hashlib.sha256(dest.read_bytes()).hexdigest(),wall_s=time.monotonic()-start)
    except Exception as e:
        return dict(row,error=repr(e),wall_s=time.monotonic()-start)

if __name__=='__main__':
    out=D/os.environ.get('NATURAL_SET','natural_candidates');(out/'references').mkdir(exist_ok=False)
    rows=json.loads((out/'manifest.json').read_text());result=[]
    with ProcessPoolExecutor(4,initializer=init) as pool:
        for row in pool.map(convert,rows):
            result.append(row);(out/'reference_manifest.json').write_text(json.dumps(result,indent=2,ensure_ascii=False));print(len(result),row['task'],row.get('error','ok'),flush=True)
