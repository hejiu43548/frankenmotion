"""Training-only exposure/curriculum; task identities never enter deployed actor."""
import torch
from dataclasses import dataclass
from joint_sampling import BroadMotion,BroadMotionCfg
class JumpExposure(BroadMotion):
 def __init__(self,cfg,env):
  super().__init__(cfg,env)
  jump=[i for i,n in enumerate(self.task_names) if n in ['jump','extra_jump']];table=self.task_names.index('table_approach');other=[i for i in range(len(self.task_names)) if i not in jump+[table]]
  n=int(self.num_envs*cfg.jump_fraction);nt=int(self.num_envs*cfg.table_fraction)
  self.env_task[:n]=torch.tensor(jump,device=self.device)[torch.arange(n,device=self.device)%len(jump)]
  self.env_task[n:n+nt]=table
  self.env_task[n+nt:]=torch.tensor(other,device=self.device)[torch.arange(self.num_envs-n-nt,device=self.device)%len(other)]
  self.takeoff_indices=torch.full_like(self.starts,-1)
  if cfg.takeoff_probability:
   feet=[self.cfg.body_names.index(x) for x in ['left_ankle_roll_link','right_ankle_roll_link']];root=self.cfg.body_names.index('pelvis')
   for k in range(len(self.starts)):
    if int(self.clip_task[k]) not in jump:continue
    lo=int(self.starts[k]);hi=int(self.ends[k]);xyz=self.motion.body_pos_w[lo:hi];air=xyz[:,feet,2].min(-1).values>.13;cross=torch.where(air[1:]&~air[:-1])[0]+1
    cross=cross[cross>=10]
    if not len(cross):continue
    a=int(cross[0]);begin=max(0,a-30);bottom=begin+int(xyz[begin:a,root,2].argmin());index=max(begin,bottom-2)
    # Only reset while both reference ankles remain near support, before extension.
    if float(xyz[index,feet,2].max())<.1:self.takeoff_indices[k]=lo+index
 def _uniform_sampling(self,env_ids):
  super()._uniform_sampling(env_ids)
  if not self.cfg.takeoff_probability:return
  candidates=self.takeoff_indices[self.clip_ids[env_ids]];chosen=(candidates>=0)&(torch.rand(len(env_ids),device=self.device)<self.cfg.takeoff_probability)
  self.time_steps[env_ids[chosen]]=candidates[chosen]
@dataclass(kw_only=True)
class JumpExposureCfg(BroadMotionCfg):
 jump_fraction:float=.4
 takeoff_probability:float=0.
 def build(self,env):return JumpExposure(self,env)
