"""Post-v1 development: source-derived task constraints on GMR, no target-Q edits."""
import sys,json,time,os
from pathlib import Path
sys.path.insert(0,'/home/pku/frankenmotion/work')
import confirmation_eval_20261003 as ce
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation
from concurrent.futures import ProcessPoolExecutor
OUT=ce.ROOT/os.environ.get('V2_OUTPUT','kick_retarget_development');OUT.mkdir(exist_ok=True)
def init():ce.init()
def refine(z,states,task,weight):
 tr=ce.g.tr;m=ce.g.M;h=z['joints_zup_m'].astype(float);side=h[0,1]-h[0,2];yaw=np.arctan2(side[1],side[0])-np.pi/2;R=Rotation.from_euler('z',-yaw).as_matrix();h=np.einsum('ij,tkj->tki',R,h);scale=tr.robot_height(m)/float(z['human_height']);d=tr.mujoco.MjData(m);prev=states[0,7:].copy();result=states.copy();lo=m.jnt_range[1:,0]+1e-5;hi=m.jnt_range[1:,1]-1e-5
 for i,state in enumerate(states):
  d.qpos[:]=state;tr.mujoco.mj_forward(m,d);p0=tr.markers(m,d);memo={};torso=(h[i,16]+h[i,17])/2-h[i,0];torso/=np.linalg.norm(torso)
  def calc(q):
   if 'q' in memo and np.array_equal(q,memo['q']):return memo['r'],memo['j']
   d.qpos[7:]=q;tr.mujoco.mj_forward(m,d);p,j=tr.markers(m,d,True);rs=[];js=[]
   if task in ['raise_hand','kick']:
    idx=21 if task=='raise_hand' else 8
    rs.append(weight*((p[idx]-p[0])-(h[i,idx]-h[i,0])*scale));js.append(weight*j[idx])
   else:
    v=(p[16]+p[17])/2-p[0];length=np.linalg.norm(v);u=v/length;rs.append(weight*(u-torso));js.append(weight*(np.eye(3)-np.outer(u,u))@((j[16]+j[17])/2)/length)
   for foot in ([7] if task=='kick' else [7,8]):rs.append(8*(p[foot]-p0[foot]));js.append(8*j[foot])
   rs.extend([.15*(q-state[7:]),.08*(q-prev)]);js.extend([.15*np.eye(29),.08*np.eye(29)]);memo.update(q=q.copy(),r=np.concatenate(rs),j=np.concatenate(js));return memo['r'],memo['j']
  fit=least_squares(lambda q:calc(q)[0],np.clip(state[7:],lo,hi),jac=lambda q:calc(q)[1],bounds=(lo,hi),max_nfev=25,ftol=1e-5,xtol=1e-5);result[i,7:]=fit.x;prev=fit.x
 return result
def run(item):
 row,w=item;tr=ce.g.tr;m=ce.g.M;stem=Path(row['path']).stem+f'_w{w}';r=dict(row,weight=w)
 try:
  z=np.load(row['path']);states=ce.g.convert(z,'uniform');states=refine(z,states,row['task'],w) if w else states;r['g1']=tr.measure(tr.get_positions(m,states),row['task'],tr.robot_height(m));arrays,fall=tr.prior_run.rollout(m,ce.g.P,*tr.upsample(states[:,7:],states[:,3:7],states[:,:3]));ts=(np.arange(len(arrays['qpos']))+1)*.02;want=1+np.arange(len(states))*.05;r.update(actual=None,fall_time=fall)
  if fall is None and ts[-1]>=want[-1]-1e-7:
   p=tr.get_positions(m,arrays['qpos']);p=np.stack([np.interp(want,ts,x) for x in p.reshape(len(p),-1).T],1).reshape(len(states),24,3);r['actual']=tr.measure(p,row['task'],tr.robot_height(m))
  np.savez_compressed(OUT/(stem+'.npz'),reference_qpos=states,time_s=ts,**arrays)
 except Exception as e:r['error']=repr(e)
 (OUT/(stem+'.json')).write_text(json.dumps(r,default=lambda x:x.item()));return r
if __name__=='__main__':
 manifest=Path(os.environ.get('V2_MANIFEST',str(ce.ROOT/'physical_development_manifest.json')));rows=[r for r in json.loads(manifest.read_text()) if r['task'] in ['kick']];weights=[int(x) for x in os.environ.get('V2_WEIGHTS','0,4,12').split(',')]
 (OUT/'protocol.json').write_text(json.dumps(dict(stage=os.environ.get('V2_STAGE','v2 development after v1 confirmation; NOT independent evidence'),source=str(manifest),weights=weights,constraints='source-derived right ankle relative position; preserve left support-foot GMR anchor; joint/time regularization',command_Q_used_in_solver=False,baseline='immutable v1 preserved'),indent=2));results=[]
 with ProcessPoolExecutor(4,initializer=init) as pool:
  for r in pool.map(run,[(r,w) for w in weights for r in rows]):results.append(r);print(len(results),r['task'],r['weight'],r.get('actual'),flush=True)
 (OUT/'results.json').write_text(json.dumps(results,indent=2,default=lambda x:x.item()))
