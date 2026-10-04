"""Descriptive development pelvis/torso yaw decomposition; no metric changes."""
import os,sys,json,argparse
from pathlib import Path
import numpy as np
import mujoco
from scipy.spatial.transform import Rotation
R=Path('/home/pku/frankenmotion');U=R/'outputs_amass/franken_unified_20261004';B=R/'outputs_amass/franken_eleven_20261003';sys.path.insert(0,str(B/'code'));os.environ['ELEVEN_OUT']=str(B)
import transfer as tr
p=argparse.ArgumentParser();p.add_argument('--name',default='joint_v5_root_short_validation');a=p.parse_args();f=U/'evaluation'/a.name;proto=json.loads((f/'protocol.json').read_text());assert proto['split']=='development_validation';rows=json.loads((f/'audited_results.json').read_text());m=tr.rt.load_model();d=mujoco.MjData(m);ids=[m.body(n).id for n in ['pelvis','torso_link']];w=int(m.jnt_qposadr[m.joint('waist_yaw_joint').id]);result=[]
def angles(states,dt,start,end):
 qs=[]
 for state in states:
  d.qpos[:]=state;mujoco.mj_forward(m,d);qs.append(d.xquat[ids].copy())
 q=np.asarray(qs);y=Rotation.from_quat(q[:,:,[1,2,3,0]].reshape(-1,4)).as_euler('ZYX')[:,0].reshape(len(states),2);y=np.unwrap(y,axis=0);t=np.arange(len(states))*dt;delta=np.asarray([np.interp(end,t,v)-np.interp(start,t,v) for v in y.T]);right=-np.arctan2(np.sin(delta),np.cos(delta));waist=-(np.interp(end,t,states[:,w])-np.interp(start,t,states[:,w]));return dict(pelvis_right_rad=float(right[0]),torso_right_rad=float(right[1]),waist_joint_right_rad=float(waist))
for r in rows:
 if r['task']!='turn':continue
 stem=Path(r['path']).stem;ref=np.load(U/proto['split']/'references'/(stem+'_uniform.npz'))['reference_qpos'];end=(len(ref)-1)*.05;x=dict(source=r['source'],command=r['command'],reference=angles(ref,.05,0,end),official_reference=r['g1'],official_actual=r['actual'])
 if r['actual'] is not None:x['actual']=angles(np.load(f/(stem+'_actual.npz'))['qpos'],.02,1,1+end)
 result.append(x)
out=U/'diagnostics';out.mkdir(exist_ok=True);dest=out/(a.name+'_turn.json');assert not dest.exists();dest.write_text(json.dumps(dict(scope='Development-only orientation decomposition. Torso and pelvis yaw differ from the established hip-marker metric under tilt; this diagnostic does not replace benchmark quantities and does not establish a causal training failure.',model=a.name,rows=result),indent=2));print(dest)
for r in result:print(r['source'],r['command'],r['reference'],r.get('actual'))
