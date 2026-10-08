import torch
from torch import nn
class BlendActor(nn.Module):
 def __init__(self,base,feedback,reference,feedback_weight:float):
  super().__init__();self.base=base;self.feedback=feedback;self.reference=reference;self.feedback_weight=feedback_weight
 def forward(self,x):
  correction=self.feedback_weight*self.feedback.physical_residual(x)+(1.-self.feedback_weight)*self.reference.physical_residual(x)
  return self.base(x)+correction@self.feedback.projection
