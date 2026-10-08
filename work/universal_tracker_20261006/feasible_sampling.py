"""One geometry-based initialization rule for all clips, without task routing.
Filters the nominal reference phase chosen for reset, not the tracked trajectory.
Original reset noise remains unchanged; this is not a stability guarantee.
"""
from dataclasses import dataclass
from pathlib import Path
import numpy as np,torch,json,hashlib
from joint_sampling import BroadMotion,BroadMotionCfg

class FeasibleBroadMotion(BroadMotion):
 def __init__(self,cfg,env):
  super().__init__(cfg,env);self.valid_phase_data=None
  if cfg.valid_reset_phases is None:return
  root=Path(cfg.valid_reset_phases);audit=json.loads((root/'audit.json').read_text());records=json.loads(Path(cfg.clip_metadata).read_text())['records'];assert len(audit['clips'])==len(records)
  for report,row in zip(audit['clips'],records):assert report['motion_path']==row['motion_path'] and report['frames']==row['frames']
  with np.load(root/'valid_phases.npz') as z:ends=z['ends'];offsets=z['valid_offsets']
  starts=np.r_[0,ends[:-1]];counts=ends-starts;assert np.all(counts>0)
  for i,row in enumerate(records):assert offsets[starts[i]:ends[i]].min()>=0 and offsets[starts[i]:ends[i]].max()<row['frames']-1
  self.valid_phase_data=(torch.tensor(offsets,device=self.device),torch.tensor(starts,device=self.device),torch.tensor(counts,device=self.device))
 def _uniform_sampling(self,env_ids):
  super()._uniform_sampling(env_ids)
  if not getattr(self,'valid_phase_data',None):return
  offsets,starts,counts=self.valid_phase_data;ids=self.clip_ids[env_ids];choice=(torch.rand(len(ids),device=self.device)*counts[ids]).long();choice[torch.rand(len(ids),device=self.device)<self.cfg.start_probability]=0;self.time_steps[env_ids]=self.starts[ids]+offsets[starts[ids]+choice]

@dataclass(kw_only=True)
class FeasibleBroadMotionCfg(BroadMotionCfg):
 valid_reset_phases:str|None=None
 def build(self,env):return FeasibleBroadMotion(self,env)
