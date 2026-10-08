"""Diagnostic only: change root translation, preserve every joint/orientation/time.
Never present these edited references as generated command-conditioned demos.
"""
import json,os
import numpy as np
from fk_conversion import D,convert
out=D/'root_translation_counterfactual';out.mkdir(exist_ok=False);rows=json.loads((D/'native_eleven_manifest.json').read_text());sources=[r for r in rows if r['task']=='walk' and r['command_index']==2];assert len(sources)==2;result=[]
for row in sources:
 ref=np.load(row['reference_path'])['reference_qpos']
 for scale in [.5,1.,1.5]:
  q=ref.copy();q[:,:2]=q[:1,:2]+scale*(q[:,:2]-q[:1,:2]);assert np.array_equal(q[:,2:],ref[:,2:]);stem=row['source']+f'_root{scale}';path=out/(stem+'.npz');os.link(row['path'],path);rp=out/(stem+'_reference.npz');mp=out/(stem+'_motion.npz');np.savez_compressed(rp,reference_qpos=q);np.savez_compressed(mp,fps=50.,**convert(q));result.append(dict(row,path=str(path),reference_path=str(rp),motion_path=str(mp),root_translation_scale=scale,scope='Counterfactual reference edit only; no generator command claim. Human motion remains unchanged; shared robot joints,orientation,timing.'))
(out/'manifest.json').write_text(json.dumps(result,indent=2))
