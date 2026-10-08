"""Development-only global timing probe, original joint path and height unchanged."""
import sys,json
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/jump_tracker_20261007';sys.path[:0]=[str(R/'work/universal_tracker_20261006'),str(R/'work'),str(R/'outputs_amass/franken_eleven_20261003/code')]
from fk_conversion import convert
out=D/'timing_probe';out.mkdir(exist_ok=True)
rows=json.loads((D/'dev_jump_manifest.json').read_text());result=[]
for row in rows:
 q=np.load(row['reference_path'])['reference_qpos']
 for speed in [.7,.85,1.15,1.3]:
  at=np.linspace(0,len(q)-1,round((len(q)-1)/speed)+1);new=np.stack([np.interp(at,np.arange(len(q)),v) for v in q.T],1);new[:,3:7]=Slerp(np.arange(len(q)),Rotation.from_quat(q[:,[4,5,6,3]]))(at).as_quat()[:,[3,0,1,2]]
  name=Path(row['path']).stem+f'_speed{speed}';human=out/(name+'.npz');human.symlink_to(row['path']);ref=out/(name+'_reference.npz');motion=out/(name+'_motion.npz');np.savez_compressed(ref,reference_qpos=new);np.savez_compressed(motion,fps=50.,**convert(new));result.append(dict(row,path=str(human),reference_path=str(ref),motion_path=str(motion),timing_speed=speed))
(out/'manifest.json').write_text(json.dumps(result,indent=2))
