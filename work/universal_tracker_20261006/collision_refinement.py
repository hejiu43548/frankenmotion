"""Common self-collision constrained arm refinement in exact simulator geometry.

Root, torso and leg coordinates are locked. Existing GMR wrist targets and arm
posture remain soft objectives. This is a disclosed retargeting ablation, not
generated motion or a new tracker, and never depends on action category.
"""
import sys,json
from pathlib import Path
import numpy as np,mujoco
sys.path.insert(0,'/home/pku/frankenmotion/work/gmr_deps')
import mink
from mink.limits.limit import Limit,Constraint
from mink.limits.collision_avoidance_limit import compute_contact_normal_jacobian
class RecoveringCollisionLimit(mink.CollisionAvoidanceLimit):
 """Displacement inequality also separates already penetrating references.
 The stock velocity damper allows zero motion at an existing penetration.
 """
 def compute_qp_inequalities(self,configuration,dt):
  G=np.zeros((len(self.geom_id_pairs),self.model.nv));h=np.full(len(G),np.inf)
  for i,(a,b) in enumerate(self.geom_id_pairs):
   distance=mujoco.mj_geomDistance(self.model,configuration.data,a,b,self.collision_detection_distance,self._fromto)
   if distance>=self.collision_detection_distance-1e-12:continue
   row=compute_contact_normal_jacobian(self.model,configuration.data,a,b,self._fromto,self._normal,self._jac1,self._jac2)
   G[i]=(-1 if distance>=0 else 1)*row;h[i]=self.gain*(distance-self.minimum_distance_from_collisions)
  return Constraint(G=G,h=h)
class FrozenDofs(Limit):
 def __init__(self,m,active):
  fixed=[i for i in range(m.nv) if i not in active];eye=np.eye(m.nv)[fixed];self.G=np.r_[eye,-eye];self.h=np.zeros(len(self.G))
 def compute_qp_inequalities(self,configuration,dt):return Constraint(G=self.G,h=self.h)
class CollisionRefiner:
 def __init__(self,scene,contract):
  self.model=m=mujoco.MjModel.from_binary_path(str(scene));self.contract=c=json.loads(Path(contract).read_text());self.qa=m.jnt_qposadr[[m.joint('robot/'+n).id for n in c['joint_names']]]
  arm_names=[n for n in c['joint_names'] if any(s in n for s in ['shoulder','elbow','wrist'])];arm_ids=[m.joint('robot/'+n).id for n in arm_names];self.arm_qa=m.jnt_qposadr[arm_ids];self.arm_limits=m.jnt_range[arm_ids];active=m.jnt_dofadr[arm_ids];self.fixed_q=np.array([i for i in range(m.nq) if i not in self.arm_qa]);self.configuration=mink.Configuration(m)
  arm_groups=[[f'robot/{side}_{part}_collision' for part in ['elbow_yaw','wrist','hand']] for side in ['left','right']];body=['robot/pelvis_collision','robot/torso_collision','robot/head_collision']+[f'robot/{side}_{part}_collision' for side in ['left','right'] for part in ['hip','thigh','shin']]
  self.collision=RecoveringCollisionLimit(m,geom_pairs=[(arm_groups[0]+arm_groups[1],body),(arm_groups[0],arm_groups[1])],gain=.5,minimum_distance_from_collisions=.003,collision_detection_distance=.05)
  self.limits=[mink.ConfigurationLimit(m),FrozenDofs(m,active),self.collision];self.posture=mink.PostureTask(m,cost=.1);self.temporal=mink.PostureTask(m,cost=.1);self.wrists=[mink.FrameTask(frame_name=f'robot/{side}_wrist_yaw_link',frame_type='body',position_cost=5.,orientation_cost=1.) for side in ['left','right']];self.tasks=[self.posture,self.temporal]+self.wrists
 def distance(self):
  m=self.model;d=self.configuration.data
  return min(mujoco.mj_geomDistance(m,d,a,b,.1,None) for a,b in self.collision.geom_id_pairs)
 def refine(self,native_states):
  output=[];logs=[];previous=np.zeros(self.model.nq)
  for frame,reference in enumerate(native_states):
   self.configuration.update(reference);before=self.distance();self.posture.set_target(reference)
   for task in self.wrists:task.set_target_from_configuration(self.configuration)
   target=reference.copy();target[self.arm_qa]=np.clip(target[self.arm_qa]+previous[self.arm_qa],self.arm_limits[:,0],self.arm_limits[:,1]);self.temporal.set_target(target);self.configuration.update(target);best=(self.distance(),target.copy());errors=[]
   for iteration in range(12):
    try:
     velocity=mink.solve_ik(self.configuration,self.tasks,.05,solver='daqp',damping=.01,limits=self.limits);self.configuration.integrate_inplace(velocity,.05)
     projected=self.configuration.data.qpos.copy();assert np.max(abs(projected[self.fixed_q]-reference[self.fixed_q]))<1e-4;projected[self.fixed_q]=reference[self.fixed_q];self.configuration.update(projected)
    except Exception as exc:errors.append(repr(exc));break
    distance=self.distance();q=self.configuration.data.qpos.copy()
    if distance>best[0]:best=(distance,q.copy())
    if distance>=.0028 and np.linalg.norm(velocity)*.05<1e-4:break
   q=self.configuration.data.qpos.copy();after=self.distance()
   if after<best[0] and after<.002:after,q=best
   fixed_error=float(np.max(abs(q[self.fixed_q]-reference[self.fixed_q])));assert fixed_error<1e-6,(frame,fixed_error,self.fixed_q[np.argmax(abs(q[self.fixed_q]-reference[self.fixed_q]))])
   q[self.fixed_q]=reference[self.fixed_q];previous=q-reference;output.append(q);logs.append(dict(frame=frame,before_distance_m=float(before),after_distance_m=float(after),max_arm_change_rad=float(np.max(abs(previous[self.arm_qa]))),solver_errors=errors))
  return np.asarray(output),logs
