"""Descriptive development jump traces; no new event gate or model selection."""
import os,sys,json,argparse
from pathlib import Path
import numpy as np
import mujoco
R=Path('/home/pku/frankenmotion');U=R/'outputs_amass/franken_unified_20261004';B=R/'outputs_amass/franken_eleven_20261003';sys.path.insert(0,str(B/'code'));os.environ['ELEVEN_OUT']=str(B)
import transfer as tr
p=argparse.ArgumentParser();p.add_argument('--name',default='joint_v4_tracking_step4000_validation');a=p.parse_args();folder=U/'evaluation'/a.name;rows=json.loads((folder/'audited_results.json').read_text());proto=json.loads((folder/'protocol.json').read_text());assert proto['split']=='development_validation';model=tr.rt.load_model();data=mujoco.MjData(model);result=[]
def describe(states,dt):
 positions=[];com=[]
 for state in states:
  data.qpos[:]=state;mujoco.mj_forward(model,data);positions.append(tr.markers(model,data));com.append(data.subtree_com[0].copy())
 p=np.asarray(positions);com=np.asarray(com);z=p[:,0,2];clear=np.minimum(p[:,7,2]-p[0,7,2],p[:,8,2]-p[0,8,2]);vz=np.gradient(com[:,2],dt);az=np.gradient(vz,dt)
 # Both ankle marker rises are a proxy only; this is not a contact-force measurement.
 mask=clear>.08;interior=mask.copy()
 for offset in [-2,-1,1,2]:interior &= np.roll(mask,offset)
 interior[:2]=False;interior[-2:]=False
 return dict(root_rise_robot_m=float(z.max()-z[0]),root_pre_peak_crouch_robot_m=float(z[0]-z[:np.argmax(z)+1].min()),peak_time_s=float(np.argmax(z)*dt),both_ankle_rise_robot_m=float(clear.max()),com_vertical_velocity_max=float(vz.max()),marker_clear_samples=int(mask.sum()),interior_clear_samples=int(interior.sum()),interior_com_vertical_accel_median=float(np.median(az[interior])) if interior.any() else None,interior_ballistic_residual_median=float(np.median(abs(az[interior]+9.81))) if interior.any() else None)
for row in rows:
 if row['task']!='jump':continue
 stem=Path(row['path']).stem;ref=np.load(U/proto['split']/'references'/(stem+'_uniform.npz'))['reference_qpos'];record=dict(source=row['source'],command=row['command'],reference=describe(ref,.05),reference_metric=row['g1'],actual_metric=row['actual'])
 if row['actual'] is not None:
  raw=np.load(folder/(stem+'_actual.npz'))['qpos'];record['actual']=describe(raw[50:],.02)
 result.append(record)
report=dict(scope='Development-only descriptive failure analysis. Kinematic reference marker clearance is not measured contact; acceleration uses finite differences in the reference robot mass model, so residuals alone do not prove dynamic infeasibility. No selection metric or threshold changed.',model=a.name,rows=result)
out=U/'diagnostics';out.mkdir(exist_ok=True);dest=out/(a.name+'_jump.json');assert not dest.exists();dest.write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
