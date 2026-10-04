"""Diagnose V1 turn timing from actual states; does not change references or metrics."""
import sys,json
from pathlib import Path
import numpy as np
B=Path('/home/pku/frankenmotion/outputs_amass/franken_eleven_20261003');N=B.parent/'franken_improve_20261003';sys.path.insert(0,str(B/'code'));import transfer as tr
m=tr.rt.load_model();rows=json.loads((N/'confirmation/candidate/results.json').read_text());rows=[r for r in rows if r['task']=='turn'];out=[]
def yaw(states):
 p=tr.get_positions(m,states);side=p[:,1,:2]-p[:,2,:2];return np.unwrap(np.arctan2(side[:,1],side[:,0]))
wrap=lambda x:np.arctan2(np.sin(x),np.cos(x))
for r in rows:
 z=np.load(N/'confirmation/candidate'/(Path(r['path']).stem+'_uniform.npz'));ref=yaw(z['reference_qpos']);act=yaw(z['qpos']);ts=z['time_s'];start=float(np.interp(1,ts,act));end=float(np.interp(6.95,ts,act));start_err=float(wrap(start-ref[0]));end_err=float(wrap(end-ref[-1]));out.append(dict(source=r['source'],command=r['command'],initial_heading_error_rad=start_err,final_heading_error_rad=end_err,turn_shortfall_rad=end_err-start_err,reference_Q=r['g1']['quantity'],actual_Q=r['actual']['quantity'],preentry_turn_rad=float(-(start-act[0]))))
summary={}
for c in sorted({r['command'] for r in out}):
 a=[r for r in out if r['command']==c];summary[str(c)]={k:float(np.mean([r[k] for r in a])) for k in ['initial_heading_error_rad','final_heading_error_rad','turn_shortfall_rad','preentry_turn_rad','reference_Q','actual_Q']}
f=N/'turn_timing_audit.json';f.write_text(json.dumps(dict(scope='post-hoc V1 diagnosis; no altered measurement window',rows=out,summary=summary),indent=2));print(json.dumps(summary,indent=2))
