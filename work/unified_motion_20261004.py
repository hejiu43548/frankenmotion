"""One reference interface for all tasks; motion IDs are sampler metadata only."""
from dataclasses import dataclass
import json
import torch
from mjlab.tasks.tracking.mdp.commands import MotionCommand,MotionCommandCfg

class MultiMotionCommand(MotionCommand):
 def __init__(self,cfg,env):
  super().__init__(cfg,env)
  data=json.load(open(cfg.clip_metadata));self.ends=torch.tensor(data['ends'],device=self.device,dtype=torch.long);self.starts=torch.cat([self.ends.new_zeros(1),self.ends[:-1]])
  assert self.ends[-1].item()==self.motion.time_step_total and torch.all(self.ends-self.starts>3)
  self.clip_ids=torch.zeros(self.num_envs,device=self.device,dtype=torch.long);self.sample_counts=torch.zeros(len(self.ends),device=self.device,dtype=torch.long)
  self.command_step_counts=torch.zeros_like(self.sample_counts)
  self.task_names=sorted({r['task'] for r in data['records']});self.clip_task=torch.tensor([self.task_names.index(r['task']) for r in data['records']],device=self.device);groups=[torch.where(self.clip_task==i)[0] for i in range(len(self.task_names))]
  self.task_sizes=torch.tensor([len(x) for x in groups],device=self.device);self.task_clips=torch.zeros((len(groups),int(self.task_sizes.max())),device=self.device,dtype=torch.long)
  for i,g in enumerate(groups):self.task_clips[i,:len(g)]=g
  self.env_task=torch.arange(self.num_envs,device=self.device)%len(groups)
 def _uniform_sampling(self,env_ids):
  if self.cfg.task_balanced_slots:
   task=self.env_task[env_ids];choice=(torch.rand(len(env_ids),device=self.device)*self.task_sizes[task]).long();ids=self.task_clips[task,choice]
  else:ids=torch.randint(len(self.ends),(len(env_ids),),device=self.device)
  self.clip_ids[env_ids]=ids
  lengths=self.ends[ids]-self.starts[ids];offset=(torch.rand(len(ids),device=self.device)*(lengths-1)).long()
  offset[torch.rand(len(ids),device=self.device)<self.cfg.start_probability]=0
  self.time_steps[env_ids]=self.starts[ids]+offset
  self.sample_counts+=torch.bincount(ids,minlength=len(self.ends))
  self.metrics['sampling_entropy'][:]=1.;self.metrics['sampling_top1_prob'][:]=1./self.task_sizes[self.env_task] if self.cfg.task_balanced_slots else 1./len(self.ends)
 def _update_command(self):
  # Episode termination/reset runs before command update. Crossing here is a bug.
  assert bool(torch.all(self.time_steps+1<self.ends[self.clip_ids])), 'Unreported reference boundary crossing'
  super()._update_command()
  self.command_step_counts+=torch.bincount(self.clip_ids,minlength=len(self.ends))

@dataclass(kw_only=True)
class MultiMotionCfg(MotionCommandCfg):
 clip_metadata:str
 start_probability:float=.35
 task_balanced_slots:bool=False
 def build(self,env):return MultiMotionCommand(self,env)

def reference_end(env):
 term=env.command_manager.get_term('motion')
 return term.time_steps>=term.ends[term.clip_ids]-1
