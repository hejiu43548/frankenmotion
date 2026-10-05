"""Offline mixed-teacher data only. Deployment student will be one actor."""
import os,sys,json,argparse,hashlib
os.environ['MUJOCO_GL']='egl'
from pathlib import Path
import numpy as np,torch,mujoco
from scipy.spatial.transform import Rotation
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/gait_demo_20261005';sys.path.insert(0,str(R/'work/g1_sim_bridge'));import g1_runtime as rt
p=argparse.ArgumentParser();p.add_argument('--count',type=int,default=8);a=p.parse_args();out=D/'teacher_data_dagger1';out.mkdir(exist_ok=False);template=D/'dev_timed/scene_000/baseline_timed';c=json.loads((template/'inference_contract.json').read_text());torch.set_num_threads(2);student=torch.jit.load(str(D/'training/distill_v1/actor_3000.pt')).eval();actor=torch.jit.load(str(R/'outputs_amass/table_demo_20261005/frozen/actor.pt')).eval();m=mujoco.MjModel.from_binary_path(str(template/'scene.mjb'));names=[m.joint(i).name.removeprefix('robot/') for i in range(1,m.njnt)];qa=np.array([int(m.joint('robot/'+n).qposadr[0]) for n in c['joint_names']]);va=np.array([int(m.joint('robot/'+n).dofadr[0]) for n in c['joint_names']]);aids=[int(np.where(m.actuator_trnid[:,0]==m.joint('robot/'+n).id)[0][0]) for n in c['action_target_names']];sonic_order=[c['joint_names'].index(n) for n in names];sonic_to_action=[names.index(n) for n in c['action_target_names']];bodyids=[i for i in range(m.nbody) if m.body(i).name.startswith('robot/')];anchor=m.body(c['anchor_body_name']).id;ar=c['reference_anchor_index'];default=np.asarray(c['default_joint_pos']);scale=np.asarray(c['action_scale']);offset=np.asarray(c['action_offset']);pol=rt.Policy();records=[r for r in json.loads((D/'training_corpus_timed/manifest.json').read_text()) if r['task']=='table_approach'];rng=np.random.default_rng(73005000);records=[records[i] for i in rng.permutation(len(records))[:a.count]];allobs=[];allactions=[];groups=[];phases=[];reports=[]
def rot(q):return Rotation.from_quat(np.asarray(q)[[1,2,3,0]])
def sensor(d,name):
 s=m.sensor(name);return d.sensordata[s.adr[0]:s.adr[0]+s.dim[0]].copy()
def observe(d,ref,i,last):
 inv=rot(d.xquat[anchor]).inv()
 def rel(k):return inv.apply(ref['body_pos_w'][k,ar]-d.xpos[anchor]),(inv*rot(ref['body_quat_w'][k,ar])).as_matrix()[:,:2].reshape(-1)
 p,o=rel(i);parts=[ref['joint_pos'][i],ref['joint_vel'][i],p,o,sensor(d,c['linear_velocity_sensor']),sensor(d,c['angular_velocity_sensor']),d.qpos[qa]-default,d.qvel[va],last]
 for dt in c['preview_offsets']:
  k=min(i+dt,len(ref['joint_pos'])-1);p,o=rel(k);parts.extend([ref['joint_pos'][k],ref['joint_vel'][k],p,o])
 return np.concatenate(parts).astype(np.float32)
for group,row in enumerate(records):
 ref=dict(np.load(row['motion_path']));scene=row['scene_metadata'];m.body_pos[m.body('demo_table').id]=[*scene['table_center'][:2],0];m.body_quat[m.body('demo_table').id]=Rotation.from_euler('z',scene['table_yaw']).as_quat()[[3,0,1,2]];d=mujoco.MjData(m);d.qpos[:3]=ref['body_pos_w'][0,0];d.qpos[3:7]=ref['body_quat_w'][0,0];d.qpos[qa]=ref['joint_pos'][0];mujoco.mj_forward(m,d);last=np.zeros(29);pol.reset(d);walkend=(21+round(row['walk_retiming_seconds']*20))*2.5;settleend=walkend+75;teacher_states=[];teacher_controls=[];mixing=[];fell=False
 with torch.inference_mode():
  for i in range(len(ref['joint_pos'])-2):
   obs=observe(d,ref,i,last);bm=actor(torch.from_numpy(obs)[None])[0].numpy();bm_target=bm*scale+offset;sonic_target,raw=pol.act(d,ref['joint_pos'][:,sonic_order],ref['joint_vel'][:,sonic_order],ref['body_quat_w'][:,0],i);u=np.clip((i-walkend)/75,0,1);alpha=u*u*(3-2*u);target=(1-alpha)*sonic_target[sonic_to_action]+alpha*bm_target;label=((target-offset)/scale).astype(np.float32);allobs.append(obs);allactions.append(label);groups.append(group);phases.append(float(alpha));student_action=student(torch.from_numpy(obs)[None])[0].numpy();target=.3*target+.7*(student_action*scale+offset);last=((target-offset)/scale).astype(np.float32);d.ctrl[aids]=target
   for _ in range(round(.02/m.opt.timestep)):mujoco.mj_step(m,d)
   mujoco.mj_forward(m,d);executed=np.array([target[c['action_target_names'].index(n)] for n in names]);sonic_raw=((executed-rt.Q0)/rt.SCALE)[rt.MJ_TO_IL];pol.update(d,sonic_raw);teacher_states.append(d.qpos.copy());teacher_controls.append(d.ctrl.copy());mixing.append(float(alpha))
   if d.qpos[2]<.35 or np.hypot(*rot(d.qpos[3:7]).as_euler('xyz')[:2])>1.05:fell=True;break
 np.savez_compressed(out/f'teacher_{group:03d}.npz',qpos=teacher_states,ctrl=teacher_controls,mixing=mixing,fps=50.);reports.append(dict(group=group,source=row,frames=len(teacher_states),fell=fell,root_final=d.qpos[:3].tolist()));(out/'reports.json').write_text(json.dumps(reports,indent=2));print(group,len(teacher_states),'fell',fell,flush=True)
np.savez_compressed(out/'dataset.npz',observations=allobs,actions=allactions,group=groups,blend=phases);(out/'protocol.json').write_text(json.dumps(dict(count=a.count,training_clips_only=True,selection_seed=73005000,scope='DAgger: execute 70% student +30% teacher joint targets in physical simulation, label with full teacher. Offline teacher: SONIC walking, smooth transition during settling to prior single interaction actor. These mixed-teacher trajectories are not final demo results. Student deployment must be separately evaluated with one network.',actor_sha256=hashlib.sha256((R/'outputs_amass/table_demo_20261005/frozen/actor.pt').read_bytes()).hexdigest()),indent=2));print('TEACHER DATA COMPLETE',len(allobs))
