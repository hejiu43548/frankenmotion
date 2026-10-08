import json,argparse,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/unified_commands_20261007';sys.path.insert(0,str(R/'outputs_amass/franken_eleven_20261003/code'))
from audit_results import native_metrics,TASKS,RANGES,TOLS,canonical
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--baseline',default='baseline_dev');a=p.parse_args();out=D/'generation'/a.name;rows=json.loads((out/'manifest.json').read_text());base=json.loads((D/'generation'/a.baseline/'manifest.json').read_text());key=lambda r:(r['source'],r['command_index']);lookup={key(r):r for r in base};result=[]
def extra_metrics(z,row):
 p=canonical(z['joints_zup_m']);root=p[:,0];side=p[:,1,:2]-p[:,2,:2];yaw=np.unwrap(np.arctan2(side[:,1],side[:,0]));k=row['kind'];c=row['command'];h=float(z['human_height']);HH=1.2701193988323212
 if k=='walk_endpoint':
  d,ang=row['extra'];target=d*np.array([np.cos(ang),np.sin(ang)]);return dict(endpoint_error_m=float(np.linalg.norm(root[-1,:2]-root[0,:2]-target)),heading_error_rad=float(abs(np.arctan2(np.sin(yaw[-1]-yaw[0]-ang),np.cos(yaw[-1]-yaw[0]-ang)))))
 if k=='turn_endpoint':return dict(angle_error_rad=float(abs(-(yaw[-1]-yaw[0])-c)),root_drift_m=float(np.linalg.norm(root[-1,:2]-root[0,:2])))
 if k=='place_hold_retract':
  wrist=p[:,21];q=float(np.mean((wrist[49:72,0]-root[49:72,0])*HH/h));return dict(held_reach_error_m=abs(q-c),held_reach_m=q,hand_lowering_m=float((np.mean(wrist[49:72,2])-np.mean(wrist[-8:,2]))*1.0486437524221748/h),root_drift_m=float(np.linalg.norm(root[-1,:2]-root[0,:2])))
 if k=='root_profile':
  raw=z['motion'];cf=z['control_features'];speed=np.linalg.norm(raw[:,1:3],axis=-1)*20;yawrate=raw[:,3]*20;sv=cf[:,20]>0;yv=cf[:,21]>0
  return dict(speed_profile_mae_m_s=float(np.mean(abs(speed[:-1]-cf[:-1,18]*3)[sv[:-1]])) if sv.any() else None,yaw_profile_mae_rad_s=float(np.mean(abs(yawrate[:-1]-cf[:-1,19]*np.pi)[yv[:-1]])) if yv.any() else None)
 return native_metrics(row)
for r in rows:
 b=lookup[key(r)];z=np.load(r['path']);bz=np.load(b['path']);p=z['joints_zup_m'];bp=bz['joints_zup_m'];err=np.linalg.norm(p-bp,axis=-1);torso=(p[:,16]+p[:,17])/2-p[:,0];torso/=np.maximum(np.linalg.norm(torso,axis=-1,keepdims=True),1e-8);step=np.degrees(np.arccos(np.clip((torso[1:]*torso[:-1]).sum(-1),-1,1)))
 row=dict(r,baseline_path=b['path'],joint_position_mae_m=float(err.mean()),max_joint_error_m=float(err.max()),max_torso_direction_step_deg=float(step.max()),finite=bool(np.isfinite(z['motion']).all()),human=extra_metrics(z,r),baseline=extra_metrics(bz,b));result.append(row)
summary={}
for k in sorted(set(r['kind'] for r in result)):
 rr=[r for r in result if r['kind']==k];s=dict(count=len(rr),mean_joint_position_error_m=float(np.mean([r['joint_position_mae_m'] for r in rr])),max_joint_error_m=max(r['max_joint_error_m'] for r in rr),torso_step_gt30=sum(r['max_torso_direction_step_deg']>30 for r in rr))
 if k in TASKS or k in ['back_departure','side_departure']:
  tid=TASKS.index(rr[0]['task']);span=RANGES[tid][1]-RANGES[tid][0]
  for field in ['baseline','human']:
   s[field]=dict(joint_pass=sum(bool(r[field]['event_pass'] and abs(r[field]['quantity']-r['command'])<=TOLS[tid]) for r in rr),event_pass=sum(bool(r[field]['event_pass']) for r in rr),semantic_E=float(np.mean([min(abs(r[field]['quantity']-r['command'])/span,1) if r[field]['event_pass'] else 1 for r in rr])))
 else:
  for field in ['baseline','human']:
   s[field]={m:float(np.mean([r[field][m] for r in rr if r[field][m] is not None])) for m in rr[0][field]}
 summary[k]=s
(out/'comparison.json').write_text(json.dumps(summary,indent=2));(out/'audited_results.json').write_text(json.dumps(result,indent=2,default=lambda x:x.item()));print(json.dumps(summary,indent=2))
