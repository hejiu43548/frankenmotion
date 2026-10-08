from pathlib import Path
import sys,json,numpy as np,mujoco
from scipy.spatial.transform import Rotation
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/temporal_mix_20261008';G=R/'outputs_amass/unified_commands_20261007/general_evaluation/temporal_mix_20261008';sys.path[:0]=[str(R/'work'),str(R/'outputs_amass/franken_eleven_20261003/code')];import transfer as tr
rows=json.loads((G/'results.json').read_text());model=tr.rt.load_model();height=tr.robot_height(model);result=[]
def measures(p,h):
 scale=tr.HH/h;side=p[:,16]-p[:,17];side/=np.maximum(np.linalg.norm(side,axis=-1,keepdims=True),1e-8);out=[]
 for sec in range(6):
  i,j=sec*20,min((sec+1)*20,len(p));seg=p[i:j];r=dict(second=sec,root_speed_equiv=float(np.linalg.norm(np.diff(seg[:,0,:2],axis=0),axis=-1).mean()*20*scale))
  for name,joint in [('left',20),('right',21)]:
   rel=p[i:j,joint]-p[i:j,0];lat=(rel*side[i:j]).sum(-1);r[name+'_height_equiv']=float(np.mean(rel[:,2])*scale);r[name+'_lateral_range_equiv']=float(np.ptp(lat)*scale)
  out.append(r)
 return out
for row in rows:
 z=np.load(row['path']);ref=np.load(row['reference_path'])['reference_qpos'];run=Path(row['run']);native=mujoco.MjModel.from_binary_path(str(run/'scene.mjb'));names=[model.joint(i).name for i in range(1,model.njnt)];order=[int(native.joint('robot/'+n).qposadr[0]) for n in names];act=np.load(run/'actual.npz')['qpos'];states=np.c_[act[:,:7],act[:,order]];p=tr.get_positions(model,states);ts=np.arange(len(p))*.02;want=1+np.arange(120)*.05;p=np.stack([np.interp(want,ts,v) for v in p.reshape(len(p),-1).T],1).reshape(120,24,3)
 result.append(dict(source=row['source'],complete=row['physical_complete'],human=measures(z['joints_zup_m'],float(z['human_height'])),g1=measures(tr.get_positions(model,ref),height),actual=measures(p,height)))
(D/'window_metrics.json').write_text(json.dumps(result,indent=2));print(json.dumps([r for r in result if r['source'].endswith('s0')],indent=2))
