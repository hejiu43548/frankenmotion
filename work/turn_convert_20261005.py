"""Native FK for reference caches, using the exact deployment body/joint order."""
import json,sys,argparse
from pathlib import Path
import numpy as np,mujoco
from scipy.spatial.transform import Rotation,Slerp
R=Path('/home/pku/frankenmotion');sys.path[:0]=[str(R/'work'),str(R/'outputs_amass/franken_eleven_20261003/code')];import transfer as tr
p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args();folder=Path(a.folder);template=R/'outputs_amass/turn_demo_20261005/development_v2/scene_000/distill_v1';c=json.loads((template/'inference_contract.json').read_text());m=mujoco.MjModel.from_binary_path(str(template/'scene.mjb'));d=mujoco.MjData(m);source=tr.rt.load_model();names=[source.joint(i).name for i in range(1,source.njnt)];order=[names.index(n) for n in c['joint_names']];qa=np.array([int(m.joint('robot/'+n).qposadr[0]) for n in c['joint_names']]);va=np.array([int(m.joint('robot/'+n).dofadr[0]) for n in c['joint_names']]);b=np.array([i for i in range(m.nbody) if m.body(i).name.startswith('robot/')]);root=m.body('robot/pelvis').id
def convert(path):
 ref=np.load(path/'reference_contact.npz')['reference_qpos'];t=np.arange(len(ref))*.05;tt=np.arange(int(np.ceil(t[-1]/.02))+1)*.02;ts=np.minimum(tt,t[-1]);pos=np.stack([np.interp(ts,t,v) for v in ref[:,:3].T],1);rots=Slerp(t,Rotation.from_quat(ref[:,[4,5,6,3]]))(ts);quat=rots.as_quat()[:,[3,0,1,2]];q=np.stack([np.interp(ts,t,v) for v in ref[:,7:].T],1)[:,order];dq=np.gradient(q,.02,axis=0);vel=np.gradient(pos,.02,axis=0);av=(rots[1:]*rots[:-1].inv()).as_rotvec()/.02;av=np.r_[av,av[-1:]];log={k:[] for k in ['joint_pos','joint_vel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w']}
 for i in range(len(q)):
  d.qpos[:7]=np.r_[pos[i],quat[i]];d.qpos[qa]=q[i];d.qvel[:6]=np.r_[vel[i],rots[i].inv().apply(av[i])];d.qvel[va]=dq[i];mujoco.mj_forward(m,d);xyz=d.xpos[b].copy();ang=d.cvel[b,:3];lin=d.cvel[b,3:]+np.cross(ang,xyz-d.subtree_com[root]);values=[q[i],dq[i],xyz,d.xquat[b],lin,ang]
  for k,v in zip(log,values):log[k].append(np.asarray(v).copy())
 return {k:np.asarray(v,dtype=np.float32) for k,v in log.items()}
test=convert(template.parent);want=np.load(template/'motion.npz');errors={k:float(np.max(abs(v-want[k]))) for k,v in test.items()};assert max(errors.values())<1e-5,errors
records=[]
for row in json.loads((folder/'manifest.json').read_text()):
 path=Path(row['source']);values=convert(path);mp=path/'motion.npz';np.savez_compressed(mp,fps=50.,**values);records.append(dict(motion_path=str(mp),scene_metadata=row,reference_metadata=json.loads((path/'reference_contact.json').read_text()),source_type='full_turn_demo'));print(row['index'],len(values['joint_pos']),flush=True)
(folder/'training_manifest.json').write_text(json.dumps(records,indent=2));(folder/'fk_parity.json').write_text(json.dumps(errors,indent=2))
