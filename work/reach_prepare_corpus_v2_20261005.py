"""Compose generated reach/lower/departure; no tabletop IK or target-dependent hand pose edits."""
import sys,json,argparse,numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation,Slerp
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/reach_demo_20261005';sys.path.insert(0,str(R/'work'));import unified_retarget_20261004 as rt
p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args();folder=Path(a.folder);rt.init();OriginalGMR=rt.g.GeneralMotionRetargeting
class HandFaithfulGMR(OriginalGMR):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  for table in [self.ik_match_table1,self.ik_match_table2]:
   for side in ['left','right']:table[side+'_wrist_yaw_link'][1]=80;table[side+'_wrist_yaw_link'][2]=5
  self.setup_retarget_configuration()
def blend(q0,q1,n):
 u=np.linspace(0,1,n+1)[1:];u=u*u*(3-2*u);q=q0[None]+u[:,None]*(q1-q0)[None];q[:,3:7]=Slerp([0,1],Rotation.from_quat(np.array([q0,q1])[:,[4,5,6,3]]))(u).as_quat()[:,[3,0,1,2]];return q
def place(q,anchor):
 q=q.copy();yaw=Rotation.from_quat(anchor[[4,5,6,3]]).as_euler('xyz')[2]-Rotation.from_quat(q[0,[4,5,6,3]]).as_euler('xyz')[2];r=Rotation.from_euler('z',yaw);offset=q[0,:3].copy();offset[2]=0;shift=anchor[:3].copy();shift[2]=0;q[:,:3]=r.apply(q[:,:3]-offset)+shift;q[:,3:7]=(r*Rotation.from_quat(q[:,[4,5,6,3]])).as_quat()[:,[3,0,1,2]];return q
def retime(q,n):
 at=np.linspace(0,len(q)-1,n);out=np.stack([np.interp(at,np.arange(len(q)),v) for v in q.T],1);out[:,3:7]=Slerp(np.arange(len(q)),Rotation.from_quat(q[:,[4,5,6,3]]))(at).as_quat()[:,[3,0,1,2]];return out
rows=[]
for record in json.loads((folder/'manifest.json').read_text()):
 dest=Path(record['source']);old=Path(record['walk_source']);base=json.loads((old/'scene.json').read_text());tag='reference_lift' if (old/'reference_lift.json').exists() else 'reference_contact';meta=json.loads((old/(tag+'.json')).read_text());states=np.load(old/(tag+'.npz'))['reference_qpos'];lo,hi=meta['segments']['walk'];newwalk=retime(states[lo:hi],round(float(np.clip(base['distance_robot_m']/.45+.8,2.8,4.8))*20));prefix=np.r_[states[:lo],newwalk,states[hi:meta['segments']['reach_hold'][0]]];delta=len(newwalk)-(hi-lo)
 rt.g.GeneralMotionRetargeting=HandFaithfulGMR;reach=rt.g.convert(np.load(dest/'human_reach.npz'),'uniform');rt.g.GeneralMotionRetargeting=OriginalGMR;reach=place(reach,prefix[-1]);lead=blend(prefix[-1],reach[0],15);start=len(prefix)+len(lead);q=np.r_[prefix,lead,reach];seg=dict(entry=meta['segments']['entry'],walk=[lo,lo+len(newwalk)],settle=[hi+delta,len(prefix)],transition_to_reach=[len(prefix),start],reach_hold=[start,start+len(reach)],reach_place_lower=[start,start+len(reach)],contact_hold=[start+48,start+72],retract_lower=[start+72,start+120]);exittime=None
 if record['exit_task']!='none':
  ex=rt.g.convert(np.load(dest/'human_exit.npz'),'uniform')
  if record['exit_task']=='sidestep':exittime=float(np.clip(record['exit_command']/.22+1.,2.5,4.2));ex=retime(ex,round(exittime*20))
  ex=place(ex,q[-1]);trans=blend(q[-1],ex[0],20);begin=len(q)+len(trans);q=np.r_[q,trans,ex];seg['exit']=[begin,len(q)]
 else:seg['exit']=[len(q),len(q)]

 if record['exit_task']!='none':
  st=q[-1].copy();st[7:]=rt.g.tr.rt.Q0;st[3:7]=Rotation.from_euler('z',Rotation.from_quat(st[[4,5,6,3]]).as_euler('xyz')[2]).as_quat()[[3,0,1,2]];dd=rt.g.tr.mujoco.MjData(rt.g.M);dd.qpos[:]=st;rt.g.tr.rt.floor_align(rt.g.M,dd);st[2]=dd.qpos[2];end=len(q);q=np.r_[q,blend(q[-1],st,30)];seg['exit_settle']=[end,len(q)]
 q=np.r_[q,np.repeat(q[-1:],30,axis=0)];assert np.array_equal(q[start:start+len(reach)],reach);c=record['command'];f=np.array([np.cos(base['direction_rad']),np.sin(base['direction_rad'])]);right=np.array([f[1],-f[0]]);target=np.array(base['goal_xy'])+(c*1.0486437524221748/1.2701193988323212+.11)*f+.17*right;scene=dict(base);scene.update(record);scene.update(reach_command_human_m=c,hand_target=[*target.tolist(),.84]);(dest/'scene.json').write_text(json.dumps(scene,indent=2));np.savez_compressed(dest/'reference_contact.npz',reference_qpos=q,fps=20.);(dest/'reference_contact.json').write_text(json.dumps(dict(segments=seg,reference_frames=len(q),fps=20,scope='Generated reach-place-lower with GMR wrist cost80/orientation5; generated back/side exit. Rigid XY/yaw placement, smooth inter-clip blends, and 1.5s final nominal-stance stopping blend. No scene-aware hand IK or table-coordinate-dependent edits to the generated reach segment. Walk and sidestep reference retiming disclosed.',walk_duration_s=len(newwalk)/20,exit_retiming_seconds=exittime,command=c,generator_sha256=record['generator_sha256']),indent=2));rows.append(scene);print('prepared',record['index'],record['exit_task'],len(q),flush=True)
(folder/'manifest.json').write_text(json.dumps(rows,indent=2))
