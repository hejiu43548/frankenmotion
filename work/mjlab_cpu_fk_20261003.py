"""CPU equivalent of mjlab's motion CSV forward-kinematics conversion."""
import numpy as np
import mujoco

def convert_cpu(env,robot,pos,quat,q,dq,vel,av,rots,path):
 model=env.sim.mj_model;data=mujoco.MjData(model);ix=robot.indexing;fq=ix.free_joint_q_adr.cpu().numpy();fv=ix.free_joint_v_adr.cpu().numpy();jq=ix.joint_q_adr.cpu().numpy();jv=ix.joint_v_adr.cpu().numpy();b=ix.body_ids.cpu().numpy();root=int(ix.root_body_id)
 log={k:[] for k in ['joint_pos','joint_vel','body_pos_w','body_quat_w','body_lin_vel_w','body_ang_vel_w']}
 for i in range(len(q)):
  data.qpos[fq]=np.r_[pos[i],quat[i]];data.qpos[jq]=q[i];data.qvel[fv]=np.r_[vel[i],rots[i].inv().apply(av[i])];data.qvel[jv]=dq[i];mujoco.mj_forward(model,data)
  xyz=data.xpos[b].copy();cvel=data.cvel[b];ang=cvel[:,:3];lin=cvel[:,3:]+np.cross(ang,xyz-data.subtree_com[root]);values=[q[i],dq[i],xyz,data.xquat[b],lin,ang]
  for key,value in zip(log,values):log[key].append(np.asarray(value).copy())
 arrays={k:np.asarray(v,dtype=np.float32) for k,v in log.items()};np.savez_compressed(path,fps=50.,**arrays);return arrays

def marker_parity(env,robot,states,source_model,tr):
 model=env.sim.mj_model;data=mujoco.MjData(model);ix=robot.indexing;fq=ix.free_joint_q_adr.cpu().numpy();jq=ix.joint_q_adr.cpu().numpy();names=[source_model.joint(i).name for i in range(1,source_model.njnt)];order=[names.index(n) for n in robot.joint_names];joint_ids=ix.joint_ids.cpu().numpy();native=[]
 for state in states:
  data.qpos[fq]=state[:7];data.qpos[jq]=state[7:][order];mujoco.mj_forward(model,data);p=np.zeros((24,3));p[0]=data.xpos[int(ix.root_body_id)]
  for index,name in tr.JOINTS.items():p[index]=data.xanchor[joint_ids[list(robot.joint_names).index(name)]]
  native.append(p)
 native=np.array(native);source=tr.get_positions(source_model,states);return float(np.max(abs(native-source)))
