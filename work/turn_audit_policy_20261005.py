"""Independent reconstruction of causal references and policy inputs from logged states."""
import argparse,json
from pathlib import Path
import numpy as np,torch,mujoco
from scipy.spatial.transform import Rotation
from turn_anchor_20261005 import anchor_reference
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--actor',required=True);a=p.parse_args();run=Path(a.run);z=np.load(run/'actual.npz');ref=dict(np.load(run/'initial_motion.npz'));c=json.loads((run/'inference_contract.json').read_text());events=json.loads((run/'anchor_events.json').read_text());eventmap={x['phase']:x for x in events};m=mujoco.MjModel.from_binary_path(str(run/'scene.mjb'));d=mujoco.MjData(m);anchor=m.body(c['anchor_body_name']).id;ar=c['reference_anchor_index'];qa=[int(m.joint('robot/'+n).qposadr[0]) for n in c['joint_names']];va=[int(m.joint('robot/'+n).dofadr[0]) for n in c['joint_names']];actor=torch.jit.load(a.actor).eval();torch.set_num_threads(2)
def rot(q):return Rotation.from_quat(np.asarray(q)[[1,2,3,0]])
def sensor(name):
 s=m.sensor(name);return d.sensordata[s.adr[0]:s.adr[0]+s.dim[0]].copy()
def observe(i,last):
 inv=rot(d.xquat[anchor]).inv()
 def rel(k):return inv.apply(ref['body_pos_w'][k,ar]-d.xpos[anchor]),(inv*rot(ref['body_quat_w'][k,ar])).as_matrix()[:,:2].reshape(-1)
 pp,rr=rel(i);parts=[ref['joint_pos'][i],ref['joint_vel'][i],pp,rr,sensor(c['linear_velocity_sensor']),sensor(c['angular_velocity_sensor']),d.qpos[qa]-c['default_joint_pos'],d.qvel[va],last]
 for dt in c['preview_offsets']:
  pp,rr=rel(min(i+dt,len(ref['joint_pos'])-1));parts.extend([ref['joint_pos'][min(i+dt,len(ref['joint_pos'])-1)],ref['joint_vel'][min(i+dt,len(ref['joint_pos'])-1)],pp,rr])
 return np.concatenate(parts).astype(np.float32)
samples=set(np.linspace(0,len(z['qpos'])-1,40,dtype=int));samples.update(i for phase in eventmap for i in range(phase-2,phase+3));errors=[];action_errors=[];event_errors=[]
with torch.inference_mode():
 for i in range(len(z['qpos'])):
  if i not in samples and i not in eventmap:continue
  d.qpos[:]=c['initial_qpos'] if i==0 else z['qpos'][i-1];d.qvel[:]=c['initial_qvel'] if i==0 else z['qvel'][i-1];d.ctrl[:]=0 if i==0 else z['ctrl'][i-1];d.qacc_warmstart[:]=0;mujoco.mj_forward(m,d)
  if i in eventmap:
   ev=anchor_reference(ref,i,d.qpos.copy(),eventmap[i]['stage']);event_errors.append(abs(ev['applied_yaw_rad']-eventmap[i]['applied_yaw_rad']))
  obs=observe(i,np.zeros(29) if i==0 else z['actions'][i-1]);act=actor(torch.from_numpy(obs)[None])[0].numpy();errors.append(float(np.max(abs(obs-z['observations'][i]))));action_errors.append(float(np.max(abs(act-z['actions'][i]))))
final=np.load(run/'motion.npz');reference_errors={k:float(np.max(abs(ref[k]-final[k]))) for k in ref};report=dict(observation_max_error=max(errors),action_max_error=max(action_errors),event_max_error=max(event_errors),reference_errors=reference_errors,samples=len(errors),events=len(events),scope='States before action independently reconstruct observations; reapply causal reference events only when command is issued. Final world reference and unchanged joint trajectories checked. No recorded future reference is used before its event.')
assert max(errors)<2e-5 and max(action_errors)<2e-5 and max(reference_errors.values())<1e-5 and max(event_errors)<1e-8,report
(run/'policy_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
