import os,sys,types,json,argparse,time,contextlib,io
from pathlib import Path
BASE=Path('/home/pku/frankenmotion/outputs_amass/franken_eleven_20261003');OUT=BASE.parent/'franken_improve_20261003'
sys.path[:0]=[str(BASE/'code'),'/home/pku/frankenmotion/work/gmr_deps'];os.environ['ELEVEN_OUT']=str(BASE)
# Import computational GMR submodule without optional viewer/Torch dependencies.
pkg=types.ModuleType('general_motion_retargeting');pkg.__path__=['/home/pku/frankenmotion/work/GMR/general_motion_retargeting'];sys.modules[pkg.__name__]=pkg
from general_motion_retargeting.motion_retarget import GeneralMotionRetargeting
import transfer as tr
from v11_20261003 import Policy
import numpy as np
from scipy.spatial.transform import Rotation
NAMES=['pelvis','left_hip','right_hip','spine1','left_knee','right_knee','spine2','left_ankle','right_ankle','spine3','left_foot','right_foot','neck','left_collar','right_collar','head','left_shoulder','right_shoulder','left_elbow','right_elbow','left_wrist','right_wrist']
M=None;P=None
def init():
 global M,P
 M=tr.rt.load_model();P=Policy(0)
def convert(z,variant):
 with contextlib.redirect_stdout(io.StringIO()):g=GeneralMotionRetargeting('smplx','unitree_g1',verbose=False)
 parents=np.load(BASE/'skeleton.npz')['parents'][:22];human=z['joints_zup_m'].astype(float)[:,:22];local=Rotation.from_rotvec(z['poses_axisangle'].reshape(-1,3)).as_matrix().reshape(-1,22,3,3)
 side=human[0,1]-human[0,2];yaw=np.arctan2(side[1],side[0])-np.pi/2;R=Rotation.from_euler('z',-yaw).as_matrix();human=np.einsum('ij,tkj->tki',R,human);human[:,:,:2]-=human[0,0,:2].copy()
 local[:,0]=R@local[:,0];global_rot=[]
 for j in range(22):global_rot.append(local[:,j] if j==0 else global_rot[parents[j]]@local[:,j])
 quat=Rotation.from_matrix(np.stack(global_rot,1).reshape(-1,3,3)).as_quat()[:,[3,0,1,2]].reshape(-1,22,4)
 if variant=='uniform':
  scale=tr.robot_height(M)/float(z['human_height']);g.human_scale_table={k:scale for k in g.human_scale_table}
 # Clip only sub-microradian numerical limit drift; retain solver constraints.
 original=g.configuration.check_limits
 def check(tol=1e-6,safety_break=True):
  c=g.configuration;adr=c._limited_qposadr;lim=c._limited_range;v=c.data.qpos[adr];b=np.maximum(lim[:,0]-v,v-lim[:,1]);small=(b>tol)&(b<=1e-5)
  if small.any():q=c.data.qpos.copy();q[adr[small]]=np.clip(v[small],lim[small,0],lim[small,1]);c.update(q)
  return original(tol=tol,safety_break=safety_break)
 g.configuration.check_limits=check
 states=[]
 for i in range(len(human)):
  frame={name:(human[i,j],quat[i,j]) for j,name in enumerate(NAMES)};states.append(g.retarget(frame,offset_to_ground=False))
 states=np.array(states);order=[g.model.joint(M.joint(j).name).qposadr[0] for j in range(1,M.njnt)];states=np.c_[states[:,:7],states[:,order]]
 d=tr.mujoco.MjData(M);d.qpos[:]=states[0];tr.rt.floor_align(M,d);states[:,2]+=d.qpos[2]-states[0,2]
 return states

def run(item):
 row,variant=item;folder=OUT/'gmr_probe';folder.mkdir(parents=True,exist_ok=True);name=Path(row['path']).stem+'_'+variant;dest=folder/(name+'.json')
 if dest.exists():return json.loads(dest.read_text())
 start=time.monotonic();r=dict(row,variant=variant)
 try:
  z=np.load(row['path']);states=convert(z,variant);q=states[:,7:];quat=states[:,3:7];root=states[:,:3];ref=tr.measure(tr.get_positions(M,states),row['task'],tr.robot_height(M));arrays,fall=tr.prior_run.rollout(M,P,*tr.upsample(q,quat,root));ts=(np.arange(len(arrays['qpos']))+1)*.02;want=1+np.arange(len(q))*.05;actual=None
  if fall is None and ts[-1]>=want[-1]-1e-7:
   ps=tr.get_positions(M,arrays['qpos']);p=np.stack([np.interp(want,ts,x) for x in ps.reshape(len(ps),-1).T],1).reshape(len(q),24,3);actual=tr.measure(p,row['task'],tr.robot_height(M))
  r.update(g1=ref,actual=actual,fall=fall);np.savez_compressed(folder/(name+'.npz'),reference_qpos=states,**arrays,time_s=ts)
 except Exception as e:r['error']=repr(e)
 r['wall_s']=time.monotonic()-start;dest.write_text(json.dumps(r,default=lambda x:x.item()));return r
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--workers',type=int,default=6);a=ap.parse_args();rows=json.loads((BASE/'generated/task_adapter_manifest.json').read_text());rows=[r for r in rows if r['source'].endswith('_p0_s0')]
 from concurrent.futures import ProcessPoolExecutor
 results=[]
 with ProcessPoolExecutor(a.workers,initializer=init) as pool:
  for r in pool.map(run,[(r,v) for v in ['stock','uniform'] for r in rows]):
   results.append(r);print(len(results),r['task'],r['variant'],r.get('error',r.get('actual')),flush=True)
 (OUT/'gmr_probe_results.json').write_text(json.dumps(results,indent=2,default=lambda x:x.item()))
