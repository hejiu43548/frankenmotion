"""Predeclared continuous transitions and simultaneous arm/locomotion references."""
from pathlib import Path
import sys,json,argparse
import numpy as np
from scipy.spatial.transform import Rotation,Slerp
R=Path('/home/pku/frankenmotion');U=R/'outputs_amass/franken_unified_20261004';B=R/'outputs_amass/franken_eleven_20261003';sys.path.insert(0,str(B/'code'))
import transfer as tr
PLANS=[('upper_sequence',['raise_hand','wave','lean']),('travel_sequence',['walk','turn','back_walk']),('dynamic_sequence',['kick','jump','reach']),('mixed_sequence',['sidestep','strike','walk'])]
def yaw(q):return Rotation.from_quat(q[[4,5,6,3]]).as_euler('ZYX')[0]
def align(ref,last):
 angle=yaw(last)-yaw(ref[0]);rot=Rotation.from_euler('z',angle);q=ref.copy();q[:,:3]=rot.apply(ref[:,:3]);shift=np.r_[last[:2]-q[0,:2],0.];q[:,:3]+=shift;rq=(rot*Rotation.from_quat(ref[:,[4,5,6,3]])).as_quat();q[:,3:7]=rq[:,[3,0,1,2]];return q,angle,shift

def canonical_component(pos,component):
 out=np.zeros_like(pos);valid=[0,*tr.JOINTS];points=pos[:,valid]-np.array(component['alignment_shift']);out[:,valid]=Rotation.from_euler('z',-component['alignment_yaw']).apply(points.reshape(-1,3)).reshape(points.shape);return out

def sequence_metrics(pos,ref,row,model,tr_module):
 refpos=tr_module.get_positions(model,ref);height=tr_module.robot_height(model);parts=[]
 for c in row['components']:
  sl=slice(c['start'],c['start']+c['frames']);p=canonical_component(pos[sl],c);rp=canonical_component(refpos[sl],c);a=tr_module.measure(p,c['task'],height);b=tr_module.measure(rp,c['task'],height);parts.append(dict(c,actual=a,reference=b))
 return dict(root_position_rmse_m=float(np.sqrt(np.mean(np.sum((pos[:,0]-refpos[:,0])**2,axis=-1)))),components=parts,all_component_events=all(c['actual']['event_pass'] for c in parts))

if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('--split',choices=['development_validation','final_test','training_extra'],required=True);a=p.parse_args();split=U/a.split;training=a.split=='training_extra';prompt_ids=range(4) if training else [0];command_ids=range(5) if training else [0,2,4]
 if a.split=='final_test':assert (U/'frozen_unified/protocol.json').exists(),'Freeze one shared checkpoint before final challenges'
 out=split/'challenges';out.mkdir(exist_ok=False);(out/'references').mkdir();manifest=json.loads((split/'manifest.json').read_text());idx={(r['task'],r['source'],r['command_index']):r for r in manifest};model=tr.rt.load_model();rows=[];maxerr=0.
 def source(task,si,ci):
  row=idx[(task,f'{task}_p{pi}_s{si}',ci)];q=np.load(split/'references'/(Path(row['path']).stem+'_uniform.npz'))['reference_qpos'];return row,q
 def save(name,si,ci,kind,ref,parts,seed):
  stem=(f'{name}_p{pi}' if training else name)+f'_s{si}_c{ci}';path=out/(stem+'.npz');np.savez_compressed(path,reference_qpos=ref);np.savez_compressed(out/'references'/(stem+'_uniform.npz'),reference_qpos=ref);rows.append(dict(task=kind,source=stem,seed=seed,command=0.,command_index=ci,path=str(path),components=parts))
 for pi in prompt_ids:
  for si in range(2):
   for ci in command_ids:
    for name,tasks in PLANS:
     blocks=[];parts=[];cursor=0
     for task in tasks:
      src,original=source(task,si,ci);q=original.copy();angle=0.;shift=np.zeros(3)
      if blocks:
       q,angle,shift=align(original,blocks[-1][-1]);last=blocks[-1][-1];alpha=np.arange(1,20)/20.;s=alpha*alpha*(3-2*alpha);bridge=last[None]+s[:,None]*(q[0]-last)[None];rot=Slerp([0.,1.],Rotation.from_quat(np.stack([last[[4,5,6,3]],q[0,[4,5,6,3]]])))(s).as_quat();bridge[:,3:7]=rot[:,[3,0,1,2]];blocks.append(bridge);cursor+=19
      c=dict(task=task,command=src['command'],start=cursor,frames=len(q),alignment_yaw=float(angle),alignment_shift=shift.tolist());restored=canonical_component(tr.get_positions(model,q),c);err=float(np.max(abs(restored-tr.get_positions(model,original))));maxerr=max(maxerr,err);assert err<1e-6
      parts.append(c);blocks.append(q);cursor+=len(q)
     save(name,si,ci,'sequence',np.concatenate(blocks),parts,src['seed'])
    for task in ['walk','back_walk']:
     src,q=source(task,si,ci);wsrc,wave=source('wave',si,ci);assert len(q)==len(wave)==120;ref=q.copy();arm=[model.jnt_qposadr[i] for i in range(1,model.njnt) if model.joint(i).name.startswith(('right_shoulder','right_elbow','right_wrist'))];assert len(arm)==7;ref[:,arm]=wave[:,arm]
     parts=[dict(task=t,command=r['command'],start=0,frames=120,alignment_yaw=0.,alignment_shift=[0.,0.,0.]) for t,r in [(task,src),('wave',wsrc)]];save(task+'_wave',si,ci,'composition',ref,parts,src['seed'])
 (out/'manifest.json').write_text(json.dumps(rows,indent=2));(out/'protocol.json').write_text(json.dumps(dict(split=a.split,sequences=sum(r['task']=='sequence' for r in rows),compositions=sum(r['task']=='composition' for r in rows),plans=PLANS,command_indices=list(command_ids),prompts=list(prompt_ids),noises=[0,1],transition='19 interior cubic blend frames, 1 second endpoint-to-endpoint; yaw/XY-align next source, no robot reset',composition='Seven right arm joints from wave; root and remaining joints from walk/back_walk; reference metrics measured rather than assumed',source_marker_transform_max_error=maxerr,inference='One checkpoint loaded once; one standing entry per complete reference; no reset at internal boundaries. A physical termination is failure.'),indent=2));print('Built',len(rows),'challenges; rigid-source marker parity',maxerr)
