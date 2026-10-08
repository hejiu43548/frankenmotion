"""Exploratory low-dimensional PD residual, optimized on training references.
A specialized jump diagnostic, NOT an all-motion policy or deployed replacement.
Original simulator, PD gains, joint constraints and torque limits remain unchanged.
"""
import numpy as np
class PulseResidual:
 def __init__(self,ref,c,parameters):
  self.p=np.array(parameters);z=ref['body_pos_w'][:,0,2];peak=np.argmax(z[50:])+50;self.bottom=np.argmin(z[50:peak+1])+50;self.amplitude=np.clip((z[peak]-z[50])/.3,.2,2.)**(self.p[11] if len(self.p)>11 else 1.);landing=np.flatnonzero(z[peak:]<=z[50]+.04);self.landing=peak+int(landing[0]) if len(landing) else len(z)-1;self.indices=[[c['action_target_names'].index(side+'_'+joint+'_joint') for side in ['left','right']] for joint in ['hip_pitch','knee','ankle_pitch']];self.log=[]
 def apply(self,d,i,aids):
  t=(i-self.bottom)*.02;centers=[-.10,.10+self.p[9],.36];widths=[.18,.12*np.exp(self.p[10]),.20];basis=np.exp(-((t-np.array(centers))/widths)**2);joint_delta=basis@self.p[:9].reshape(3,3)*self.amplitude;joint_delta=joint_delta+(np.exp(-(((i-self.landing)*.02)/.18)**2)*self.p[12:15]*self.amplitude if len(self.p)>=15 else 0);delta=np.zeros(len(aids))
  for ids,value in zip(self.indices,joint_delta):delta[ids]=value
  delta=np.clip(delta,-.8,.8);d.ctrl[aids]+=delta;self.log.append(delta)
