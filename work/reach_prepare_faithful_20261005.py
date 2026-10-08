"""Rigid placement and short pose transitions only; no scene-aware arm IK."""
import sys,json,argparse,numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation,Slerp
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/reach_demo_20261005';sys.path.insert(0,str(R/'work'));import unified_retarget_20261004 as rt
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--export',required=True);p.add_argument('--exit',choices=['none','back_walk','sidestep'],default='none');a=p.parse_args();out=D/a.name;out.mkdir(exist_ok=False);rt.init();OriginalGMR=rt.g.GeneralMotionRetargeting
class HandFaithfulGMR(OriginalGMR):
 def __init__(self,*args,**kwargs):
  super().__init__(*args,**kwargs)
  for table in [self.ik_match_table1,self.ik_match_table2]:
   for side in ['left','right']:
    table[side+'_wrist_yaw_link'][1]=80
    table[side+'_wrist_yaw_link'][2]=5
  self.setup_retarget_configuration()
rt.g.GeneralMotionRetargeting=HandFaithfulGMR
m=rt.g.M;d=rt.g.tr.mujoco.MjData(m);hand=m.joint('right_wrist_yaw_joint').bodyid[0];src=R/'outputs_amass/gait_demo_20261005/dev_timed/scene_000';base=json.loads((src/'scene.json').read_text());meta=json.loads((src/'reference_contact.json').read_text());prefix=np.load(src/'reference_contact.npz')['reference_qpos'][:meta['segments']['reach_hold'][0]];rows=[]
def blend(q0,q1,n):
 u=np.linspace(0,1,n+1)[1:];u=u*u*(3-2*u);q=q0[None]+u[:,None]*(q1-q0)[None];q[:,3:7]=Slerp([0,1],Rotation.from_quat(np.array([q0,q1])[:,[4,5,6,3]]))(u).as_quat()[:,[3,0,1,2]];return q
def place(q,anchor):
 q=q.copy();yaw=Rotation.from_quat(anchor[[4,5,6,3]]).as_euler('xyz')[2]-Rotation.from_quat(q[0,[4,5,6,3]]).as_euler('xyz')[2];r=Rotation.from_euler('z',yaw);offset=q[0,:3].copy();offset[2]=0;shift=anchor[:3].copy();shift[2]=0;q[:,:3]=r.apply(q[:,:3]-offset)+shift;q[:,3:7]=(r*Rotation.from_quat(q[:,[4,5,6,3]])).as_quat()[:,[3,0,1,2]];return q
for row in json.loads((Path(a.export)/'manifest.json').read_text()):
 c=row['command'];reach=rt.g.convert(np.load(row['path']),'uniform');reach=place(reach,prefix[-1]);lead=blend(prefix[-1],reach[0],15);start=len(prefix)+len(lead);q=np.r_[prefix,lead,reach];segments=dict(meta['segments']);segments['reach_hold']=[start,start+len(reach)];segments['reach_place_lower']=[start,start+len(reach)];segments['contact_hold']=[start+48,start+72];segments['retract_lower']=[start+72,start+120];segments['transition_to_reach']=[len(prefix),start]
 if a.exit!='none':
  ex=np.load(D/f'command_probe/{a.exit}_p0_c0_g1.npz')['reference_qpos'];ex=place(ex,q[-1]);trans=blend(q[-1],ex[0],20);exstart=len(q)+len(trans);q=np.r_[q,trans,ex];segments['exit']=[exstart,len(q)]
 else:segments['exit']=[len(q),len(q)]
 q=np.r_[q,np.repeat(q[-1:],30,axis=0)];scene=dict(base);scene.update(index=row['index'],source=str(out/f'scene_{row["index"]:03d}'),reach_command_human_m=c,exit_task=a.exit,hand_target=[base['goal_xy'][0]+c*1.0486437524221748/1.2701193988323212+.11,-.17,.84]);dest=Path(scene['source']);dest.mkdir();(dest/'scene.json').write_text(json.dumps(scene,indent=2));np.savez_compressed(dest/'reference_contact.npz',reference_qpos=q,fps=20.);rm=dict(meta);rm.update(segments=segments,reference_frames=len(q),reach_source=row['path'],scope='Generated reach-place-lower with GMR wrist position cost80/orientation5 (no scene input), rigidly aligned and blended for transition; no arm IK, no target-dependent pose edits.',command=c);(dest/'reference_contact.json').write_text(json.dumps(rm,indent=2));palm=[]
 for state in q:
  d.qpos[:]=state;rt.g.tr.mujoco.mj_forward(m,d);palm.append(d.xpos[hand]+d.xmat[hand].reshape(3,3)@np.array([.11,.01,0.]))
 palm=np.asarray(palm);np.savez_compressed(dest/'reference_diagnostics.npz',palm=palm,qpos=q);hold=palm[start+48:start+72];rm['generated_g1_hold_palm_mean']=hold.mean(0).tolist();rm['generated_g1_hold_palm_std']=hold.std(0).tolist();(dest/'reference_contact.json').write_text(json.dumps(rm,indent=2));rows.append(scene);print(row['index'],c,rm['generated_g1_hold_palm_mean'],rm['generated_g1_hold_palm_std'],flush=True)
(out/'manifest.json').write_text(json.dumps(rows,indent=2))
