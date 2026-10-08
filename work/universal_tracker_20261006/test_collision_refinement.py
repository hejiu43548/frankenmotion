import json
import numpy as np,mujoco
from collision_refinement import CollisionRefiner
from fk_conversion import D,model,jq,order
p=D/'table_evaluation/baseline_stable/scene_000/reference_input';r=CollisionRefiner(p/'scene.mjb',p/'inference_contract.json');rows=json.loads((D/'native_eleven_manifest.json').read_text());row=next(x for x in rows if x['task']=='walk');q=np.load(row['reference_path'])['reference_qpos'];n=np.repeat(model.qpos0[None],len(q),0);n[:,:7]=q[:,:7];n[:,jq]=q[:,7:][:,order];checks=[]
for state in n[::20]:
 r.configuration.update(state);constraint=r.collision.compute_qp_inequalities(r.configuration,.05)
 for k,(a,b) in enumerate(r.collision.geom_id_pairs):
  d=mujoco.mj_geomDistance(r.model,r.configuration.data,a,b,.1,None)
  if d>=-.002:continue
  analytic=-constraint.G[k];numeric=np.zeros(r.model.nv);epsilon=1e-5
  for j in range(r.model.nv):
   velocity=np.zeros(r.model.nv);velocity[j]=1;plus=state.copy();minus=state.copy();mujoco.mj_integratePos(r.model,plus,velocity,epsilon);mujoco.mj_integratePos(r.model,minus,velocity,-epsilon);r.configuration.update(plus);dp=mujoco.mj_geomDistance(r.model,r.configuration.data,a,b,.1,None);r.configuration.update(minus);dm=mujoco.mj_geomDistance(r.model,r.configuration.data,a,b,.1,None);numeric[j]=(dp-dm)/(2*epsilon)
  r.configuration.update(state);error=float(np.max(abs(numeric-analytic)));checks.append(dict(distance=d,jacobian_max_error=error));assert error<1e-3,checks[-1]
assert checks
(D/'collision_jacobian_test.json').write_text(json.dumps(checks,indent=2));print(len(checks),max(x['jacobian_max_error'] for x in checks))
