"""Training-only source balancing, with no source/task identity in observations."""
import torch
from dataclasses import dataclass
from table_training_20261005 import TableMotion,TableMotionCfg
class BroadMotion(TableMotion):
 def __init__(self,cfg,env):
  super().__init__(cfg,env)
  extras=[i for i,n in enumerate(self.task_names) if n.startswith('extra_')]
  if not extras:return
  table=self.task_names.index('table_approach');old=[i for i in range(len(self.task_names)) if i not in extras and i!=table]
  assert old and 0<cfg.new_motion_fraction<1-cfg.table_fraction
  nt=int(self.num_envs*cfg.table_fraction);ne=int(self.num_envs*cfg.new_motion_fraction)
  self.env_task[:nt]=table
  self.env_task[nt:nt+ne]=torch.tensor(extras,device=self.device)[torch.arange(ne,device=self.device)%len(extras)]
  self.env_task[nt+ne:]=torch.tensor(old,device=self.device)[torch.arange(self.num_envs-nt-ne,device=self.device)%len(old)]
@dataclass(kw_only=True)
class BroadMotionCfg(TableMotionCfg):
 new_motion_fraction:float=.25
 def build(self,env):return BroadMotion(self,env)
