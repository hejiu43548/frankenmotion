"""Measure generated backward and side-step stages in their initial pelvis-heading frame."""
import json,sys
from pathlib import Path
import numpy as np,mujoco
from scipy.spatial.transform import Rotation
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/universal_tracker_20261006';sys.path[:0]=[str(R/'work'),str(R/'outputs_amass/franken_eleven_20261003/code')];import transfer as tr
base=D/'general_evaluation/frozen_demo_retreat';out=json.loads((base/'command_metrics.json').read_text());source=tr.rt.load_model();height=tr.robot_height(source)
for r in json.loads((base/'results.json').read_text()):
 run=Path(r['run']);a=np.load(run/'actual.npz');m=mujoco.MjModel.from_binary_path(str(run/'scene.mjb'));order=[m.jnt_qposadr[m.joint('robot/'+source.joint(i).name).id] for i in range(1,source.njnt)];q=np.c_[a['qpos'][:,:7],a['qpos'][:,order]]
 for stage in r['segments']:
  if stage['task'] not in ['back_walk','sidestep']:continue
  row=next(v for v in out['results'] if v['stage']==stage['stage'] and v['route']==stage['route'])
  if not row['complete']:continue
  pos=tr.get_positions(source,q);t=np.arange(len(pos))*.02;want=1+np.arange(stage['start20'],stage['end20'])*.05;assert want[-1]<=t[-1];angles=np.unwrap(Rotation.from_quat(q[:,[4,5,6,3]]).as_euler('xyz')[:,2]);yaw=float(np.interp(want[0],t,angles));pp=np.stack([np.interp(want,t,v) for v in pos.reshape(len(pos),-1).T],1).reshape(-1,24,3);rot=Rotation.from_euler('z',-yaw);pp=rot.apply((pp-pp[:1,:1]).reshape(-1,3)).reshape(pp.shape);row['actual']=tr.measure(pp,stage['task'],height);row['source_times_s']=[float(want[0]),float(want[-1])];row['measurement_scope']='Exact original 20Hz source-time sampling of composed stage in initial pelvis-heading coordinates; human-equivalent units. Stage transitions are excluded. Diagnostic, not an additional benchmark request.'
out['scope']+=' Backward/sidestep quantities are measured in each stage initial pelvis-heading coordinates; stage-local diagnostic, separate from frozen standalone benchmark.';(base/'command_metrics.json').write_text(json.dumps(out,indent=2,default=lambda v:v.item()));print(json.dumps(out,indent=2,default=lambda v:v.item()))
