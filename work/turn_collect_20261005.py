"""Training-only two-teacher rollout/DAgger. Deployed student has no router."""
import json,argparse,os
os.environ['MUJOCO_GL']='egl'
from pathlib import Path
import numpy as np,torch,mujoco
from scipy.spatial.transform import Rotation
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/turn_demo_20261005';OLD=R/'outputs_amass/gait_demo_20261005';NEW=R/'outputs_amass/reach_demo_20261005'
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--student');p.add_argument('--beta',type=float,default=.7);a=p.parse_args();out=D/a.name;out.mkdir(exist_ok=False);torch.set_num_threads(2)
template=NEW/'final_paired16/scene_011/selected';c=json.loads((template/'inference_contract.json').read_text());m=mujoco.MjModel.from_binary_path(str(template/'scene.mjb'));actors=[torch.jit.load(str(x/'frozen/actor.pt')).eval() for x in [OLD,NEW]];student=torch.jit.load(a.student).eval() if a.student else None
qa=np.array([int(m.joint('robot/'+n).qposadr[0]) for n in c['joint_names']]);va=np.array([int(m.joint('robot/'+n).dofadr[0]) for n in c['joint_names']]);aids=[int(np.where(m.actuator_trnid[:,0]==m.joint('robot/'+n).id)[0][0]) for n in c['action_target_names']];anchor=m.body(c['anchor_body_name']).id;ar=c['reference_anchor_index'];default=np.array(c['default_joint_pos']);scale=np.array(c['action_scale']);offset=np.array(c['action_offset'])
def rot(q):return Rotation.from_quat(np.asarray(q)[[1,2,3,0]])
def sensor(d,name):
 s=m.sensor(name);return d.sensordata[s.adr[0]:s.adr[0]+s.dim[0]].copy()
def observe(d,ref,i,last):
 inv=rot(d.xquat[anchor]).inv()
 def rel(k):return inv.apply(ref['body_pos_w'][k,ar]-d.xpos[anchor]),(inv*rot(ref['body_quat_w'][k,ar])).as_matrix()[:,:2].reshape(-1)
 pp,oo=rel(i);parts=[ref['joint_pos'][i],ref['joint_vel'][i],pp,oo,sensor(d,c['linear_velocity_sensor']),sensor(d,c['angular_velocity_sensor']),d.qpos[qa]-default,d.qvel[va],last]
 for dt in c['preview_offsets']:
  k=min(i+dt,len(ref['joint_pos'])-1);pp,oo=rel(k);parts.extend([ref['joint_pos'][k],ref['joint_vel'][k],pp,oo])
 return np.concatenate(parts).astype(np.float32)
rows=[r for r in json.loads((NEW/'corpus_v7/manifest.json').read_text()) if r.get('source_type')=='generated_reach_lower_exit_no_scene_ik'];assert len(rows)==48
obsall=[];actall=[];groups=[];blends=[];reports=[]
for group,row in enumerate(rows):
 ref=dict(np.load(row['motion_path']));scene=row['scene_metadata'];seg=row['reference_metadata']['segments'];end=min(len(ref['joint_pos'])-2,int(seg['reach_hold'][1]*2.5)+10);m.body_pos[m.body('demo_table').id]=[*scene['table_center'][:2],0];m.body_quat[m.body('demo_table').id]=Rotation.from_euler('z',scene['table_yaw']).as_quat()[[3,0,1,2]];d=mujoco.MjData(m);d.qpos[:3]=ref['body_pos_w'][0,0];d.qpos[3:7]=ref['body_quat_w'][0,0];d.qpos[qa]=ref['joint_pos'][0];mujoco.mj_forward(m,d);last=np.zeros(29);fell=False;states=[]
 with torch.inference_mode():
  for i in range(end):
   obs=observe(d,ref,i,last);x=torch.from_numpy(obs)[None];old,new=[act(x)[0].numpy() for act in actors];u=np.clip((i-seg['walk'][1]*2.5)/75,0,1);alpha=u*u*(3-2*u);label=((1-alpha)*old+alpha*new).astype(np.float32);obsall.append(obs);actall.append(label);groups.append(group);blends.append(alpha);last=label.copy()
   if student is not None:last=(1-a.beta)*label+a.beta*student(x)[0].numpy()
   if c['clip_actions'] is not None:last=np.clip(last,-c['clip_actions'],c['clip_actions'])
   d.ctrl[aids]=last*scale+offset
   for _ in range(round(.02/m.opt.timestep)):mujoco.mj_step(m,d)
   mujoco.mj_forward(m,d);states.append(d.qpos.copy())
   if d.qpos[2]<.35 or np.hypot(*rot(d.qpos[3:7]).as_euler('xyz')[:2])>1.05:fell=True;break
 np.savez_compressed(out/f'rollout_{group:03d}.npz',qpos=states,fps=50.);reports.append(dict(group=group,source=row['motion_path'],frames=len(states),fell=fell));(out/'reports.json').write_text(json.dumps(reports,indent=2));print(group,len(states),fell,flush=True)
np.savez_compressed(out/'dataset.npz',observations=obsall,actions=actall,group=groups,blend=blends);(out/'protocol.json').write_text(json.dumps(dict(training_only=True,teachers=[str(x/'frozen/actor.pt') for x in [OLD,NEW]],student=a.student,student_fraction=a.beta if student else 0,validation_groups=list(range(42,48)),reference_source='Existing training split only; no final paired16 observations or labels used; template supplies identical model and observation contract only.',deployment='Must separately evaluate one fixed student actor.'),indent=2))
