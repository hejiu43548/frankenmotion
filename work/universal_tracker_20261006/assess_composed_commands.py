"""Verify locomotion plus wave event in generated composition demos.
Pointing remains a visual semantic check; do not equate survival to task success.
"""
import sys,json,argparse
from pathlib import Path
import numpy as np,mujoco
R=Path('/home/pku/frankenmotion');sys.path[:0]=[str(R/'work'),str(R/'outputs_amass/franken_eleven_20261003/code')]
import transfer as tr
from audit_results import canonical
p=argparse.ArgumentParser();p.add_argument('--run',required=True);a=p.parse_args();out=Path(a.run);rows=json.loads((out/'results.json').read_text());m=tr.rt.load_model();rh=tr.robot_height(m);native=mujoco.MjModel.from_binary_path(str(out/'scene.mjb'));order=[int(native.jnt_qposadr[native.joint('robot/'+m.joint(i).name).id]) for i in range(1,m.njnt)];result=[]
for row in rows:
 human=np.load(row['path']);ref=np.load(row['reference_path'])['reference_qpos'];h=canonical(human['joints_zup_m']);g=tr.get_positions(m,ref);record=dict(source=row['source'],command=row['command'],task=row['task'],physical_complete=row['physical_complete'],human_walk=tr.measure(h,'walk',float(human['human_height'])),g1_walk=tr.measure(g,'walk',rh))
 if row['task']=='walk_wave':record.update(human_wave=tr.measure(h,'wave',float(human['human_height'])),g1_wave=tr.measure(g,'wave',rh))
 if row['physical_complete']:
  z=np.load(Path(row['run'])/'actual.npz');states=np.c_[z['qpos'][:,:7],z['qpos'][:,order]];positions=tr.get_positions(m,states);ts=np.arange(len(states))*.02;want=1+np.arange(len(ref))*.05;pos=np.stack([np.interp(want,ts,x) for x in positions.reshape(len(states),-1).T],1).reshape(-1,24,3);record['actual_walk']=tr.measure(pos,'walk',rh)
  if row['task']=='walk_wave':record['actual_wave']=tr.measure(pos,'wave',rh)
 result.append(record)
(out/'composed_command_metrics.json').write_text(json.dumps(dict(scope=__doc__,results=result),indent=2,default=lambda x:x.item()));print(json.dumps(result,indent=2,default=lambda x:x.item()))
