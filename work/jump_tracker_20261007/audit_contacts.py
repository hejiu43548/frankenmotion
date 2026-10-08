"""Independent geometry/contact check on saved physical states, not a new success definition."""
import argparse,json
from pathlib import Path
import numpy as np,mujoco
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();root=Path(a.run);rows=json.load(open(root/'audited_results.json'));out=[]
m=mujoco.MjModel.from_binary_path(str(root/'scene.mjb'));d=mujoco.MjData(m);floors={i for i in range(m.ngeom) if m.geom_type[i]==mujoco.mjtGeom.mjGEOM_PLANE};left={i for i in range(m.ngeom) if (m.geom(i).name or '').startswith('robot/left_foot')};right={i for i in range(m.ngeom) if (m.geom(i).name or '').startswith('robot/right_foot')};assert floors and left and right
for row in rows:
 if row['task']!='jump':continue
 if 'error' in row:out.append(dict(source=row['source'],command=row['command'],error=row['error']));continue
 z=np.load(Path(row['run'])/'actual.npz');flags=[]
 for q in z['qpos']:
  d.qpos[:]=q;mujoco.mj_forward(m,d);touch=set()
  for c in d.contact:
   if c.dist<=.001:
    pair=set(map(int,c.geom));
    if pair&floors:touch|=pair-floors
  flags.append((bool(touch&left),bool(touch&right),bool(touch-left-right)))
 flags=np.array(flags);air=~flags.any(1);air[:50]=False;edges=np.diff(np.r_[False,air,False].astype(int));starts=np.where(edges==1)[0];ends=np.where(edges==-1)[0];peak=int(np.argmax(z['qpos'][50:,2]))+50 if len(z['qpos'])>50 else 0;segments=[(int(s),int(e)) for s,e in zip(starts,ends) if s<=peak<e];seg=max(segments,key=lambda x:x[1]-x[0]) if segments else None;duration=(seg[1]-seg[0])*.02 if seg else 0.;after=flags[seg[1]:] if seg else np.empty((0,3),bool);landed=bool(len(after) and np.any(after[:,:2].all(1)));strict=bool(row['physical_complete'] and duration>=.06 and landed)
 out.append(dict(source=row['source'],seed=row.get('seed'),command=row['command'],physical_complete=row['physical_complete'],apex_frame=peak,apex_air_segment=seg,apex_air_duration_s=duration,bilateral_foot_contact_after_flight=landed,complete_air_and_land=strict,nonfoot_ground_contact_frames=int(flags[50:,2].sum()),run=row['run']))
summary=dict(scope='Contact geometry reconstructed with mj_forward from saved physical qpos at50Hz. A contact is distance<=1mm; includes foot and nonfoot floor contact. Independent check only, frozen primary metric unchanged. Air interval must contain observed pelvis apex, last>=60ms, and later show bilateral foot contact.',requests=len(out),complete_air_and_land=sum(r.get('complete_air_and_land',False) for r in out),results=out);(root/'contact_audit.json').write_text(json.dumps(summary,indent=2));print({k:v for k,v in summary.items() if k not in ['results','scope']})
