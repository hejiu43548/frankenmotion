"""Shared tracker training utilities: table placement on reset, global hand reward."""
import json,torch
from dataclasses import dataclass
from unified_motion_20261004 import MultiMotionCommand,MultiMotionCfg
class TableMotion(MultiMotionCommand):
 def __init__(self,cfg,env):
  super().__init__(cfg,env)
  records=json.load(open(cfg.clip_metadata))['records'];tables=[];quats=[]
  import math
  for r in records:
   s=r.get('scene_metadata')
   if s:tables.append([*s['table_center'][:2],0]);quats.append([math.cos(s['table_yaw']/2),0,0,math.sin(s['table_yaw']/2)])
   else:tables.append([100.,100.,0.]);quats.append([1.,0.,0.,0.])
  self.table_pos=torch.tensor(tables,device=self.device);self.table_quat=torch.tensor(quats,device=self.device);self.table_id=env.sim.mj_model.body('demo_table').id
  table_task=self.task_names.index('table_approach');other=[i for i in range(len(self.task_names)) if i!=table_task];half=int(self.num_envs*cfg.table_fraction);self.env_task[:half]=table_task;self.env_task[half:]=torch.tensor(other,device=self.device)[torch.arange(self.num_envs-half,device=self.device)%len(other)]
 def _resample_command(self,env_ids):
  super()._resample_command(env_ids)
  if hasattr(self,'table_pos') and getattr(self._env,'table_fields_ready',False):
   ids=self.clip_ids[env_ids];self._env.sim.model.body_pos[env_ids,self.table_id]=self.table_pos[ids]+self._env.scene.env_origins[env_ids];self._env.sim.model.body_quat[env_ids,self.table_id]=self.table_quat[ids]
@dataclass(kw_only=True)
class TableMotionCfg(MultiMotionCfg):
 table_fraction:float=.5
 def build(self,env):return TableMotion(self,env)
def hands_global(env,std=.08):
 term=env.command_manager.get_term('motion');ids=[term.cfg.body_names.index(x) for x in ['left_wrist_yaw_link','right_wrist_yaw_link']];error=(term.body_pos_w[:,ids]-term.robot_body_pos_w[:,ids]).square().sum(-1)
 return torch.exp(-error/(std*std)).mean(-1)
def table_scene(spec):
 import mujoco
 b=spec.worldbody.add_body(name='demo_table',pos=[100,100,0]);b.add_geom(name='table_top',type=mujoco.mjtGeom.mjGEOM_BOX,size=[.4,.48,.035],pos=[0,0,.765],rgba=[.5,.28,.12,1],contype=1,conaffinity=1,friction=[.8,.02,.002])
 for i,x in enumerate([-.32,.32]):
  for j,y in enumerate([-.4,.4]):b.add_geom(name=f'table_leg_{i}{j}',type=mujoco.mjtGeom.mjGEOM_BOX,size=[.025,.025,.365],pos=[x,y,.365],contype=1,conaffinity=1)
