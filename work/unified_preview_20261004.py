"""Shared reference lookahead; contains no task identity or policy routing."""
import torch
from mjlab.managers.observation_manager import ObservationTermCfg
from mjlab.utils.lab_api.math import subtract_frame_transforms,matrix_from_quat
OFFSETS=(5,10,20)  # 50 Hz: 0.1, 0.2, 0.4 seconds.
LONG_OFFSETS=(5,10,20,35,50)
PREVIEW_DIM=3*(29+29+3+6)
def reference_preview(env,offsets=OFFSETS):
 t=env.command_manager.get_term('motion');end=t.ends[t.clip_ids]-1 if hasattr(t,'ends') else torch.full_like(t.time_steps,t.motion.time_step_total-1);parts=[]
 for offset in offsets:
  idx=torch.minimum(t.time_steps+offset,end).clamp_min(0);pos=t.motion.body_pos_w[idx,t.motion_anchor_body_index]+env.scene.env_origins;quat=t.motion.body_quat_w[idx,t.motion_anchor_body_index]
  relpos,relquat=subtract_frame_transforms(t.robot_anchor_pos_w,t.robot_anchor_quat_w,pos,quat);ori=matrix_from_quat(relquat)[...,:2].reshape(env.num_envs,6)
  parts.extend([t.motion.joint_pos[idx],t.motion.joint_vel[idx],relpos,ori])
 return torch.cat(parts,-1)
def configure_preview(cfg,offsets=OFFSETS):
 for key in ['actor','critic']:cfg.observations[key].terms['reference_preview']=ObservationTermCfg(func=reference_preview,params={"offsets":tuple(offsets)})
 return cfg

def checkpoint_offsets(path):
 x=torch.load(path,map_location="cpu",weights_only=False);legacy="model_state_dict" in x;state=x["model_state_dict"] if legacy else x["actor_state_dict"];dim=state["actor.0.weight" if legacy else "mlp.0.weight"].shape[1]
 mapping={160:(),361:OFFSETS,495:LONG_OFFSETS};assert dim in mapping, f"Unsupported shared actor input dimension {dim}"
 return mapping[dim]
