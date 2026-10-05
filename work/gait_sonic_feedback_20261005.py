"""Frozen SONIC on the identical native BM model, observation contract and joint targets; no switching."""
import os,sys,argparse,json,hashlib,shutil
os.environ['MUJOCO_GL']='egl'
from pathlib import Path
import numpy as np,mujoco
from scipy.optimize import least_squares
R=Path('/home/pku/frankenmotion');sys.path.insert(0,str(R/'work/g1_sim_bridge'));import g1_runtime as rt
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--output',required=True);p.add_argument('--hand-feedback',action='store_true');a=p.parse_args();src=Path(a.run);out=Path(a.output);out.mkdir(exist_ok=False);c=json.loads((src/'inference_contract.json').read_text());result=json.loads((src/'result.json').read_text());scene=result['scene'];m=mujoco.MjModel.from_binary_path(str(src/'scene.mjb'));d=mujoco.MjData(m);legacy=rt.load_model();names=[m.joint(i).name.removeprefix('robot/') for i in range(1,m.njnt)];assert names==[legacy.joint(i).name for i in range(1,legacy.njnt)];ref=np.load(src/'motion.npz');order=[c['joint_names'].index(n) for n in names];q=ref['joint_pos'][:,order];dq=ref['joint_vel'][:,order];quat=ref['body_quat_w'][:,0];aids=[]
for n in names:
 j=m.joint('robot/'+n).id;ids=np.where(m.actuator_trnid[:,0]==j)[0];assert len(ids)==1;aids.append(int(ids[0]))
d.qpos[:]=c['initial_qpos'];d.qvel[:]=c['initial_qvel'];mujoco.mj_forward(m,d);pol=rt.Policy();pol.reset(d);hand=m.geom('robot/right_hand_collision').id;table=m.geom('table_top').id;pelvis=m.body('robot/pelvis').id;records={k:[] for k in ['qpos','qvel','ctrl','actions','phases','root','palm']};contacts=[];termination=None
arm=np.array([j for j,n in enumerate(names) if n.startswith('right_') and any(k in n for k in ['shoulder','elbow','wrist'])]);armqa=7+arm;limits=m.jnt_range[1:][arm];original_q=q.copy();correction=np.zeros(len(arm));dd=mujoco.MjData(m);wanted=[];meta=json.loads((src.parent/'reference_contact.json').read_text());reach_start=round(meta['segments']['reach_hold'][0]*2.5)
for i in range(len(q)):
 dd.qpos[:3]=ref['body_pos_w'][i,0];dd.qpos[3:7]=quat[i];dd.qpos[7:]=q[i];mujoco.mj_kinematics(m,dd);wanted.append(dd.geom_xpos[hand].copy())
wanted=np.array(wanted);feedback=[]
for i in range(len(q)-2):
 if a.hand_feedback and i>=reach_start and i%5==0:
  state=d.qpos.copy();nominal=original_q[i,arm];previous=nominal+correction
  def residual(x):
   dd.qpos[:]=state;dd.qpos[armqa]=x;mujoco.mj_kinematics(m,dd);return np.r_[15*(dd.geom_xpos[hand]-wanted[i]),.04*(x-nominal),.1*(x-previous)]
  fit=least_squares(residual,np.clip(previous,limits[:,0]+1e-4,limits[:,1]-1e-4),bounds=(limits[:,0]+1e-4,limits[:,1]-1e-4),max_nfev=30);correction+=np.clip(fit.x-nominal-correction,-.08,.08);q[i:,arm]=original_q[i:,arm]+correction;dq=np.gradient(q,.02,axis=0);feedback.append(dict(frame=i,correction=correction.tolist()))
 target,act=pol.act(d,q,dq,quat,i);d.ctrl[aids]=target
 for _ in range(round(.02/m.opt.timestep)):mujoco.mj_step(m,d)
 mujoco.mj_forward(m,d);pol.update(d,act);values=[d.qpos.copy(),d.qvel.copy(),d.ctrl.copy(),act.copy(),i,d.xpos[pelvis].copy(),d.geom_xpos[hand].copy()]
 for k,v in zip(records,values):records[k].append(v)
 found=[]
 for j in range(d.ncon):
  con=d.contact[j]
  if hand in con.geom and table in con.geom:
   force=np.zeros(6);mujoco.mj_contactForce(m,d,j,force);top=bool(con.pos[2]>scene['table_top']-.012 and abs(con.frame[2])>.8 and d.geom_xpos[hand,2]>scene['table_top']);found.append(dict(distance=float(con.dist),normal_force=float(force[0]),position=con.pos.tolist(),top_surface=top))
 contacts.append(found);w,x,y,z=d.qpos[3:7];roll=np.arctan2(2*(w*x+y*z),1-2*(x*x+y*y));pitch=np.arcsin(np.clip(2*(w*y-z*x),-1,1))
 if d.xpos[pelvis,2]<.35 or np.hypot(roll,pitch)>1.05:termination=(i+1)*.02;break
np.savez_compressed(out/'actual.npz',**records,fps=50.);flags=[any(x['top_surface'] and x['normal_force']>.2 for x in row) for row in contacts];longest=current=0
for flag in flags:current=current+1 if flag else 0;longest=max(longest,current)
sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();hashes={n:sha(rt.POLICY/n) for n in ['model_encoder.onnx','model_decoder.onnx']};cksha=hashlib.sha256(json.dumps(hashes,sort_keys=True).encode()).hexdigest();result.update(backend='native MuJoCo CPU with frozen SONIC; identical BM scene/position actuators',checkpoint='SONIC fixed encoder+decoder, no task routing',checkpoint_sha256=cksha,sonic_hashes=hashes,single_tracker=True,physical_complete=termination is None,termination_time=termination,duration_s=len(flags)*.02,reference_phase_final=len(flags),root_final=records['root'][-1].tolist(),palm_final=records['palm'][-1].tolist(),goal_error_m=float(np.linalg.norm(records['root'][-1][:2]-scene['goal_xy'])),palm_target_error_m=float(np.linalg.norm(records['palm'][-1]-scene['hand_target'])),hand_contact_s=sum(flags)*.02,longest_hand_contact_s=longest*.02)
result['hand_feedback']=a.hand_feedback;result['reference_feedback']='Scene-aware right-arm IK using actual current torso/root state at 10Hz; no physical state overwrite';(out/'feedback.json').write_text(json.dumps(feedback));result['success']=bool(result['physical_complete'] and result['goal_error_m']<.2 and result['palm_target_error_m']<.1 and result['longest_hand_contact_s']>=1.)
(out/'result.json').write_text(json.dumps(result,indent=2));(out/'contacts.json').write_text(json.dumps(contacts));
for n in ['scene.mjb','motion.npz','inference_contract.json']:shutil.copy2(src/n,out/n)
print(result['success'],result['goal_error_m'],result['palm_target_error_m'],result['longest_hand_contact_s'],flush=True)
