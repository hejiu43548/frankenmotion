"""Experimental shared-policy + model-based vertical impulse feedback.
No external force; corrections pass through original PD targets and actuator limits.
Only reference preview (<=1s), proprioception and nominal contact state are used.
Not a trained unified-network result; CPU privileged-contact diagnostic.
"""
import numpy as np,mujoco
class ImpulseFeedback:
 def __init__(self,m,ref,c,qa,va,aids,strength):
  self.m=m;self.strength=strength;self.va=va;self.aids=aids;self.root=m.body('robot/pelvis').id;self.mass=m.body_subtreemass[self.root];self.feet=[m.body('robot/'+n).id for n in ['left_ankle_roll_link','right_ankle_roll_link']]
  self.com=[];self.feetz=[];d=mujoco.MjData(m)
  for i in range(len(ref['joint_pos'])):
   d.qpos[:7]=np.r_[ref['body_pos_w'][i,0],ref['body_quat_w'][i,0]];d.qpos[qa]=ref['joint_pos'][i];mujoco.mj_kinematics(m,d);mujoco.mj_comPos(m,d);self.com.append(d.subtree_com[self.root].copy());self.feetz.append(d.xpos[self.feet,2].min())
  self.com=np.array(self.com);self.feetz=np.array(self.feetz);self.v=np.gradient(self.com,.02,axis=0);self.log=[]
 def apply(self,d,i):
  m=self.m;mujoco.mj_subtreeVel(m,d);now=d.subtree_com[self.root,2];v=d.subtree_linvel[self.root,2];h=self.com[i:min(i+51,len(self.com)),2].max();soon=self.feetz[i:min(i+16,len(self.feetz))].max();contact=[]
  for foot in self.feet:
   if any((m.geom_bodyid[d.contact[k].geom1]==foot or m.geom_bodyid[d.contact[k].geom2]==foot) and d.contact[k].dist<.002 for k in range(d.ncon)):contact.append(foot)
  force=0.;delta=np.zeros(len(self.aids));active=bool(contact and soon>.13 and self.v[i,2]>.15 and h-now>.03)
  if active:
   desired=np.sqrt(2*9.81*max(h-now,0));force=np.clip(self.mass*self.strength*(desired-v),0,2*self.mass*9.81)
   torque=np.zeros(m.nv)
   for foot in contact:
    jac=np.zeros((3,m.nv));mujoco.mj_jacBody(m,d,jac,None,foot);torque-=jac[2]*force/len(contact)
   delta=np.clip(torque[self.va]/m.actuator_gainprm[self.aids,0],-.5,.5);d.ctrl[self.aids]+=delta
  self.log.append([float(active),float(force),float(now),float(v),float(h),float(np.max(abs(delta)))])
