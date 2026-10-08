import json,sys,argparse
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/unified_generator_20261007';sys.path.insert(0,str(R/'outputs_amass/franken_eleven_20261003/code'))
from audit_results import native_metrics,stats,TASKS,RANGES,TOLS
p=argparse.ArgumentParser();p.add_argument('--name',required=True);a=p.parse_args();out=D/'generation'/a.name;rows=json.loads((out/'manifest.json').read_text());result=[]
for r in rows:
 z=np.load(r['path']);p=z['joints_zup_m'];v=(p[:,16]+p[:,17])/2-p[:,0];v/=np.maximum(np.linalg.norm(v,axis=-1,keepdims=True),1e-9);hip=p[:,1]-p[:,2];yy=hip-(hip*v).sum(-1,keepdims=True)*v;yy/=np.maximum(np.linalg.norm(yy,axis=-1,keepdims=True),1e-9);rot=Rotation.from_matrix(np.stack([np.cross(yy,v),yy,v],-1));step=np.degrees((rot[1:]*rot[:-1].inv()).magnitude());tilt=np.degrees(np.arccos(np.clip(v[:,2],-1,1)));yaw=np.unwrap(np.arctan2(hip[:,1],hip[:,0]));human=native_metrics(r)
 result.append(dict(r,human=human,max_torso_step_deg=float(step.max()),initial_tilt_deg=float(tilt[0]),max_pelvis_drop_m=float(p[0,0,2]-p[:,0,2].min()),max_root_speed_m_s=float(np.linalg.norm(np.diff(p[:,0],axis=0),axis=-1).max()*20),yaw_range_deg=float(np.degrees(np.ptp(yaw))),net_yaw_deg=float(np.degrees(yaw[-1]-yaw[0]))))
summary={}
for i,t in enumerate(TASKS):
 rr=[r for r in result if r['task']==t];span=RANGES[i][1]-RANGES[i][0];s=stats(rr,'human',span,TOLS[i]);s['semantic_E_all']=float(np.mean([min(abs(r['human']['quantity']-r['command'])/span,1) if r['human']['event_pass'] else 1 for r in rr]));s['torso_step_gt30']=sum(r['max_torso_step_deg']>30 for r in rr);s['initial_tilt_gt90']=sum(r['initial_tilt_deg']>90 for r in rr);s['root_speed_gt10']=sum(r['max_root_speed_m_s']>10 for r in rr);s['yaw_range_gt360']=sum(r['yaw_range_deg']>360 for r in rr);summary[t]=s
(out/'audited_results.json').write_text(json.dumps(result,indent=2,default=lambda x:x.item()));(out/'summary.json').write_text(json.dumps(summary,indent=2,default=lambda x:x.item()));agg=dict(requests=len(result),joint_pass=sum(v['joint_pass'] for v in summary.values()),event_pass=sum(v['event_pass'] for v in summary.values()),macro_semantic_E_all=float(np.mean([s['semantic_E_all'] for s in summary.values()])));(out/'audit.json').write_text(json.dumps(agg,indent=2));print(json.dumps(agg),flush=True)
