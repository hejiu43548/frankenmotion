"""Reference-derived flight intent; no task ID, height command or phase label input."""
import torch

def data(env):
 t=env.command_manager.get_term('motion');root=t.cfg.body_names.index('pelvis');feet=[t.cfg.body_names.index(n) for n in ['left_ankle_roll_link','right_ankle_roll_link']]
 idx=torch.minimum(t.time_steps+10,t.ends[t.clip_ids]-1);future=t.motion.body_pos_w[idx][:,feet,2].min(-1).values
 now=t.body_pos_w[:,feet,2].min(-1).values-env.scene.env_origins[:,2]
 mask=((torch.maximum(now,future)>.13)).to(now.dtype)
 return t,root,feet,mask

def vertical_height(env):
 t,r,f,mask=data(env);e=t.body_pos_w[:,r,2]-t.robot_body_pos_w[:,r,2]
 return mask*(torch.exp(-(e/.15).square())-.25*torch.nn.functional.huber_loss(e,torch.zeros_like(e),delta=.15,reduction='none')/.15)

def vertical_velocity(env):
 t,r,f,mask=data(env);e=t.body_lin_vel_w[:,r,2]-t.robot_body_lin_vel_w[:,r,2]
 return mask*(torch.exp(-(e/.8).square())-.2*torch.nn.functional.huber_loss(e,torch.zeros_like(e),delta=.8,reduction='none')/.8)

def simultaneous_clearance(env):
 t,r,f,mask=data(env);target=t.body_pos_w[:,f,2].min(-1).values;actual=t.robot_body_pos_w[:,f,2].min(-1).values
 return mask*torch.exp(-((target-actual)/.1).square())
