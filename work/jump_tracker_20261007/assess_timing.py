"""Diagnostic only: jump extrema/landing metric extended to retimed clips.
No fixed-time-window or velocity metric is changed; only jump may differ in length.
"""
import sys,json,runpy
from pathlib import Path
import numpy as np
R=Path('/home/pku/frankenmotion');W=R/'work/jump_tracker_20261007';sys.path[:0]=[str(R/'work'),str(R/'outputs_amass/franken_eleven_20261003/code')]
import transfer as tr
original=tr.measure
def measure(p,task,height):
 if task!='jump' or len(p)==60:return original(p,task,height)
 assert p.shape[1:]==(24,3);scale=tr.HH/height;z=p[:,0,2];quantity=float(z.max()-z[0]);clear=np.minimum(p[:,7,2]-p[0,7,2],p[:,8,2]-p[0,8,2]);peak=int(np.argmax(clear));landed=bool(np.any(clear[peak:]<=.08/scale))
 result=dict(quantity=quantity*scale,both_foot_clearance_m=float(clear[peak]),landed=landed,event_pass=bool(quantity>=.18/scale and clear[peak]>=.12/scale and landed))
 if len(p)<60:
  # Duplicating the final sample cannot change any jump maximum/landing equation.
  parity=original(np.concatenate([p,np.repeat(p[-1:],60-len(p),axis=0)]),task,height)
  assert abs(parity['quantity']-result['quantity'])<1e-10 and parity['event_pass']==result['event_pass'] and parity['landed']==result['landed']
 return result
tr.measure=measure
runpy.run_path(str(W/'assess.py'),run_name='__main__')
name=sys.argv[sys.argv.index('--name')+1];path=R/'outputs_amass/jump_tracker_20261007/general_evaluation'/name/'eleven_audit.json';data=json.loads(path.read_text());data['scope']=__doc__;data['caution']='Retargeted diagnostic references differ from fixed main benchmark. Jump uses identical extrema/landing equations on variable duration; other tasks use untouched frozen metric. Do not pool with fresh final benchmark.';path.write_text(json.dumps(data,indent=2))
