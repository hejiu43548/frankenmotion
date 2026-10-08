"""Predeclared one-factor simulator robustness diagnostics, not hardware claims."""
import numpy as np,mujoco
PROFILES=('nominal','friction_0p6','mass_1p1','delay_20ms','lateral_push_40N')
def apply_model_profile(model,profile):
 assert profile in PROFILES
 if profile=='friction_0p6':model.geom_friction[:,0]*=.6
 if profile=='mass_1p1':
  ids=[i for i in range(model.nbody) if (model.body(i).name or '').startswith('robot/')];model.body_mass[ids]*=1.1;model.body_inertia[ids]*=1.1;mujoco.mj_setConst(model,mujoco.MjData(model))
def description(profile):
 return dict(nominal='Unchanged nominal native model',friction_0p6='Tangential friction coefficient multiplied by0.6',mass_1p1='Robot body masses and inertias multiplied by1.1',delay_20ms='One control-tick delay of actuator position target; previous policy action observation remains commanded action',lateral_push_40N='40N world-Y pelvis force from simulation2.00s to2.10s')[profile]
