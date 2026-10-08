"""Task-independent causal filtering of control correction around the reference.

An inference ablation, not a claim that filtering is learned or category specific.
alpha=1 exactly bypasses all arithmetic for a strict baseline parity check.
"""
import numpy as np

class ReferenceResidualFilter:
 def __init__(self,contract,alpha):
  assert 0<alpha<=1
  self.alpha=alpha;self.scale=np.asarray(contract['action_scale']);self.offset=np.asarray(contract['action_offset'])
  self.order=[contract['joint_names'].index(n) for n in contract['action_target_names']]
  self.previous=None
 def apply(self,action,reference):
  if self.alpha==1:return action.copy()
  nominal=np.asarray(reference)[self.order];residual=action*self.scale+self.offset-nominal
  if self.previous is None:self.previous=residual.copy()
  filtered=self.alpha*residual+(1-self.alpha)*self.previous;self.previous=filtered.copy()
  return (nominal+filtered-self.offset)/self.scale
