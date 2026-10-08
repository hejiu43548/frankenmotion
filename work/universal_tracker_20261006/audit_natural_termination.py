import json,argparse
from pathlib import Path
import numpy as np
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();out=Path(a.run);rows=json.loads((out/'results.json').read_text());results=[]
def tilt(q):
 w,x,y,z=q;return float(np.hypot(np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y)),np.arcsin(np.clip(2*(w*y-z*x),-1,1)))*180/np.pi)
for row in rows:
 if 'error' in row:continue
 r=Path(row['run']);q=np.load(r/'actual.npz')['qpos'];ref=np.load(r/'motion.npz');t=len(q)-1;planned_low=ref['body_pos_w'][:,0,2]<.35;ref_tilt=np.asarray([tilt(x) for x in ref['body_quat_w'][:,0]]);results.append(dict(task=row['task'],uid=row.get('uid'),physical_complete=row['physical_complete'],last_actual_height_m=float(q[-1,2]),last_actual_tilt_deg=tilt(q[-1,3:7]),same_frame_reference_height_m=float(ref['body_pos_w'][t,0,2]),same_frame_reference_tilt_deg=float(ref_tilt[t]),reference_ever_below_0p35=bool(planned_low.any()),reference_ever_tilt_over_60=bool((ref_tilt>60).any()),reference_min_height_m=float(ref['body_pos_w'][:,0,2].min()),same_frame_root_error_m=float(np.linalg.norm(q[-1,:3]-ref['body_pos_w'][t,0]))))
(out/'termination_audit.json').write_text(json.dumps(results,indent=2))
for task in sorted({r['task'] for r in results}):
 rr=[r for r in results if r['task']==task];print(task,'fails',sum(not r['physical_complete'] for r in rr),'reference_low',sum(r['reference_ever_below_0p35'] for r in rr),'reference_tilt',sum(r['reference_ever_tilt_over_60'] for r in rr),'failed_low_ref',[(r['uid'],round(r['last_actual_height_m'],2),round(r['same_frame_reference_height_m'],2),round(r['last_actual_tilt_deg']),round(r['same_frame_reference_tilt_deg'])) for r in rr if not r['physical_complete']])
