"""Convert the fixed long-horizon 20Hz reference to the standard 50Hz native interface."""
import json,hashlib
from pathlib import Path
import numpy as np
from fk_conversion import convert
D=Path('/home/pku/frankenmotion/outputs_amass/universal_tracker_20261006');out=D/'long_horizon_sequence';rows=json.loads((out/'manifest.json').read_text())
for r in rows:
 p=out/'four_cycles_motion.npz';np.savez_compressed(p,fps=50.,**convert(np.load(r['reference_path'])['reference_qpos']));r.update(motion_path=str(p),motion_sha256=hashlib.sha256(p.read_bytes()).hexdigest())
(out/'native_manifest.json').write_text(json.dumps(rows,indent=2))
