"""Export any FrankenMotion [T,205] 20Hz motion NPZ for migrate.py, using the generator environment."""
import argparse,sys,os
from pathlib import Path
import numpy as np
import torch
ap=argparse.ArgumentParser(description=__doc__)
ap.add_argument('--input',type=Path,required=True);ap.add_argument('--output',type=Path,required=True)
ap.add_argument('--franken-root',type=Path,default=Path('/home/pku/frankenmotion'))
args=ap.parse_args();source=args.input.resolve();out=args.output.resolve();root=args.franken_root.resolve()
if out.exists():raise FileExistsError(out)
data=np.load(source,allow_pickle=False)['motion']
if data.ndim!=2 or data.shape[1]!=205 or not np.isfinite(data).all():raise ValueError('Expected finite [T,205] motion')
sys.path[:0]=[str(root),str(root/'work/franken_sim_bridge')];os.chdir(root)
from src.tools.inference import load_smplh
from bridge import decode_fk,export_source
torch.set_num_threads(2);out.parent.mkdir(parents=True,exist_ok=True)
with torch.no_grad():export_source(decode_fk(torch.from_numpy(data).float(),load_smplh()),out)
print(out)
