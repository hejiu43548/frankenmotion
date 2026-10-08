"""Reconstruct post-step actuator forces from recorded state and applied control.
Nominal MuJoCo model diagnostic; not measured hardware torque.
"""
import argparse,json
from pathlib import Path
import numpy as np,mujoco
p=argparse.ArgumentParser();p.add_argument('--run',required=True);p.add_argument('--tasks',nargs='+',default=['jump','strike','walk']);a=p.parse_args();root=Path(a.run);assert not (root/'INVALIDATED.json').exists();m=mujoco.MjModel.from_binary_path(str(root/'scene.mjb'));d=mujoco.MjData(m);assert np.all(m.actuator_forcelimited);limits=np.max(abs(m.actuator_forcerange),axis=1);names=[m.actuator(i).name for i in range(m.nu)];results=[]
for row in json.loads((root/'results.json').read_text()):
 if row['task'] not in a.tasks or 'error' in row:continue
 z=np.load(Path(row['run'])/'actual.npz');forces=[]
 for i in range(50,len(z['qpos'])):
  d.qpos[:]=z['qpos'][i];d.qvel[:]=z['qvel'][i];d.ctrl[:]=z['ctrl'][i-1];mujoco.mj_forward(m,d);forces.append(d.actuator_force.copy())
 f=np.array(forces);saturation=abs(f)>=.99*limits;per=saturation.mean(0);top=np.argsort(-per)[:6];results.append(dict(task=row['task'],command=row['command'],source=row['source'],physical_complete=row['physical_complete'],frames=len(f),fraction_frames_any_actuator_saturated=float(saturation.any(-1).mean()),mean_fraction_actuators_saturated=float(saturation.mean()),most_saturated=[dict(name=names[i],fraction=float(per[i]),limit_Nm=float(limits[i]),peak_absolute_torque_Nm=float(np.max(abs(f[:,i])))) for i in top]))
(root/'actuation_audit.json').write_text(json.dumps(dict(scope='Reconstructed nominal-model actuator force from recorded post-step qpos/qvel and applied ctrl. Saturation threshold99% actuator forcerange. Does not establish hardware feasibility or prove saturation causes command error.',results=results),indent=2))
for r in results:print(r['task'],r['command'],round(r['fraction_frames_any_actuator_saturated'],3),r['most_saturated'][:2])
