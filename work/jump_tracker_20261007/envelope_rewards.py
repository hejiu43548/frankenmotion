"""Reference apex objective, allowing a ballistic trajectory between key frames.
Task-agnostic gate derives from two-foot flight in the same <=1s reference horizon.
This is an inspired pilot, not an implementation or reproduction of AdaMimic.
"""
import torch
from flight_rewards import data

def targets(env):
 t,r,f,mask=data(env);off=torch.arange(51,device=t.device);idx=torch.minimum(t.time_steps[:,None]+off[None],t.ends[t.clip_ids,None]-1)
 z=t.motion.body_pos_w[idx][:,:,r,2]+env.scene.env_origins[:,None,2]
 apex=z.max(-1).values
 soon=t.motion.body_pos_w[torch.minimum(t.time_steps+15,t.ends[t.clip_ids]-1)][:,f,2].min(-1).values>.13
 enabled=(mask.bool()|soon).to(z.dtype)
 return t,r,apex,enabled

def predicted_apex(env):
 t,r,apex,enabled=targets(env);z=t.robot_body_pos_w[:,r,2];v=t.robot_body_lin_vel_w[:,r,2].clamp(min=0,max=5);estimate=z+v.square()/(2*9.81);e=estimate-apex
 return enabled*(torch.exp(-(e/.12).square())-.5*torch.nn.functional.huber_loss(e,torch.zeros_like(e),delta=.12,reduction='none')/.12)

class RelaxDuringFlight:
 def __init__(self,original,factor=.2):self.original=original;self.factor=factor
 def __call__(self,env,**kwargs):
  t,r,f,mask=data(env)
  return self.original(env,**kwargs)*(1-mask*(1-self.factor))
