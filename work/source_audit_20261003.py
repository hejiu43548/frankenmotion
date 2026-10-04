import os,sys,json
from pathlib import Path
BASE=Path('/home/pku/frankenmotion/outputs_amass/franken_eleven_20261003');OUT=BASE.parent/'franken_improve_20261003'
sys.path.insert(0,str(BASE/'code'));os.environ['ELEVEN_OUT']=str(BASE)
import transfer as tr
import numpy as np
from scipy.spatial.transform import Rotation
m=tr.rt.load_model();rows=json.loads((BASE/'generated/task_adapter_manifest.json').read_text());result=[]
for r in rows:
    z=np.load(r['path']);p=z['joints_zup_m'].astype(float);ref=np.load(BASE/'simulation/task_adapter'/(Path(r['path']).stem+'_reference.npz'))
    g=tr.get_positions(m,np.c_[ref['root'],ref['quat'],ref['q']]);d={k:r[k] for k in ['task','source','command','command_index']}
    for key,ps,height in [('human',p,float(z['human_height'])),('g1',g,tr.robot_height(m))]:
        side=ps[0,1]-ps[0,2];yaw=np.arctan2(side[1],side[0])-np.pi/2;ps=Rotation.from_euler('z',-yaw).apply(ps.reshape(-1,3)).reshape(ps.shape)
        feet=ps[:,[7,8]];v=np.gradient(feet,.05,axis=0);floor=np.quantile(feet[:,:,2],.05);mask=feet[:,:,2]<floor+.04
        rv=np.gradient(ps[:,0,:2],.05,axis=0)
        low=np.argmin(feet[:,:,2],axis=1);idx=np.arange(len(feet));support_v=v[idx,low,:2];support_height=feet[idx,low,2]
        bodyrelative=feet-ps[:,:1];bodyv=np.gradient(bodyrelative,.05,axis=0)
        d[key]=dict(low_foot_xy_speed=float(np.mean(np.linalg.norm(support_v,axis=-1))),near_floor_xy_speed=float(np.mean(np.linalg.norm(v[:,:,:2],axis=-1)[mask])),root_path_speed=float(np.mean(np.linalg.norm(rv,axis=-1))),root_x_speed=float(np.mean(rv[:,0])),leg_relative_x_range=float(np.mean(np.ptp(bodyrelative[:,:,0],axis=0))),root_z_range=float(np.ptp(ps[:,0,2])),low_foot_z_range=float(np.ptp(support_height)),root_displacement=(ps[-1,0]-ps[0,0]).tolist(),height=height)
    q=ref['q'];d['joint_speed_max']=float(abs(np.gradient(q,.05,axis=0)).max());d['joint_acceleration_p95']=float(np.quantile(abs(np.gradient(np.gradient(q,.05,axis=0),.05,axis=0)),.95))
    result.append(d)
(OUT/'source_contact_audit_canonical.json').write_text(json.dumps(result,indent=2))
for task in ['walk','back_walk','sidestep','jump','wave','lean']:
    print(task,flush=True)
    for ci in [0,4]:
        a=[r for r in result if r['task']==task and r['command_index']==ci]
        print(ci,{k:round(float(np.mean([r['human'][k] for r in a])),3) for k in ['low_foot_xy_speed','root_path_speed','leg_relative_x_range','root_z_range','low_foot_z_range']},flush=True)
