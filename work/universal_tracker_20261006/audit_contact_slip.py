"""Tangential contact-point speed for geometrically touching foot/ground pairs.
This is a geometric proxy, not a force-weighted hardware slip measurement.
"""
import sys,json,argparse
from pathlib import Path
import numpy as np,mujoco
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--sonic',action='store_true');p.add_argument('--table',action='store_true');p.add_argument('--segment',choices=['walk','away']);a=p.parse_args();root=Path(a.run);rows=[] if a.table else json.loads((root/('audited_results.json' if a.sonic else 'results.json')).read_text());results=[]
if a.table:
 rows=json.loads((root/'summary.json').read_text())['results'];rows=[dict(r,source='scene_'+str(r['scene']['index']),task='table',command=r['scene']['reach_command_human_m']) for r in rows]
if a.sonic:
 sys.path.insert(0,'/home/pku/frankenmotion/work/g1_sim_bridge');import g1_runtime as rt
 m=rt.load_model()
else:m=mujoco.MjModel.from_binary_path(str((Path(rows[0]['run']) if a.table else root)/'scene.mjb'))
ground={int(m.geom_bodyid[g]) for g in range(m.ngeom) if m.geom_type[g]==mujoco.mjtGeom.mjGEOM_PLANE and m.body_weldid[int(m.geom_bodyid[g])]==0};assert ground
d=mujoco.MjData(m);feet={i for i in range(m.nbody) if 'ankle_roll_link' in (m.body(i).name or '')};jac=np.zeros((3,m.nv));jr=np.zeros_like(jac)
for row in rows:
 z=np.load(Path(row['run'])/'actual.npz');values=[];frames=0
 lo,hi=50,len(z['qpos'])
 if a.table and a.segment:
  meta=json.loads((Path(row['run']).parent/'reference_contact.json').read_text());segment='walk' if a.segment=='walk' else 'exit';lo,hi=[int(x*2.5) for x in meta['segments'][segment]];hi=min(hi,len(z['qpos']))
 for i in range(lo,hi):
  d.qpos[:]=z['qpos'][i];d.qvel[:]=z['qvel'][i];mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);mujoco.mj_collision(m,d);speeds=[]
  for j in range(d.ncon):
   c=d.contact[j];b1,b2=[int(m.geom_bodyid[g]) for g in c.geom]
   if c.dist>.001:continue
   foot=b1 if b1 in feet and b2 in ground else b2 if b2 in feet and b1 in ground else None
   if foot is None:continue
   mujoco.mj_jac(m,d,jac,jr,c.pos,foot);v=jac@d.qvel;normal=c.frame[:3];speeds.append(float(np.linalg.norm(v-v.dot(normal)*normal)))
  if speeds:values.append(float(np.mean(speeds)));frames+=1
 results.append(dict(source=row['source'],task=row['task'],command=row.get('command'),root_translation_scale=row.get('root_translation_scale'),physical_complete=row['physical_complete'],contact_frames=frames,observed_phase=a.segment or 'all_after_entry',observed_frames=hi-lo,mean_contact_tangent_speed_m_s=float(np.mean(values)) if values else None,p95_contact_tangent_speed_m_s=float(np.quantile(values,.95)) if values else None))
(root/(('contact_slip_'+a.segment if a.segment else 'contact_slip_audit')+'.json')).write_text(json.dumps(dict(scope=__doc__,rows=results),indent=2));print(json.dumps(results,indent=2))
