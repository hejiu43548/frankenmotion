import sys,json,hashlib
from pathlib import Path
import numpy as np
R=Path('/home/pku/frankenmotion');U=R/'outputs_amass/universal_tracker_20261006';D=R/'outputs_amass/jump_tracker_20261007';sys.path[:0]=[str(R/'work/universal_tracker_20261006')]
from fk_conversion import convert
rows=json.load(open(U/'joint_corpus_expanded_corrected/manifest.json'));test=json.load(open(U/'natural_test_motion/manifest.json'));test_sources={r['source'] for r in test};rows=[r for r in rows if r['task'].startswith('extra_')];assert not {r['source'] for r in rows}&test_sources
out=D/'natural_retention';out.mkdir(exist_ok=False);(out/'motions').mkdir();(out/'links').mkdir();result=[]
for i,row in enumerate(rows):
 r=dict(row);name=f'{i:04d}_'+Path(r['path']).stem;link=out/'links'/(name+'.npz');link.symlink_to(r['path']);motion=out/'motions'/(name+'.npz');q=np.load(r['reference_path'])['reference_qpos'];np.savez_compressed(motion,fps=50.,**convert(q,entry=True));r.update(path=str(link),motion_path=str(motion),pulse_parameters=[0.]*15,command=None);result.append(r)
(out/'manifest.json').write_text(json.dumps(result,indent=2));(out/'protocol.json').write_text(json.dumps(dict(requests=len(result),purpose='Baseline-policy on-policy states on official training-only mocap, with zero residual labels. Source-disjoint from natural regression test; partial finite rollouts retained for preservation, not counted as successful demonstrations.'),indent=2))
