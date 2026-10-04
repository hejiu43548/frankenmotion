from pathlib import Path
import sys,os,json,hashlib
import numpy as np
BASE=Path('/home/pku/frankenmotion/outputs_amass/franken_eleven_20261003');ROOT=BASE.parent/'franken_improve_20261003';sys.path.insert(0,str(BASE/'code'));os.environ['ELEVEN_OUT']=str(BASE)
from audit_results import stats,native_metrics
import transfer as tr
p=json.loads((ROOT/'frozen_kick_v3/protocol.json').read_text());assert hashlib.sha256(Path(p['selected_generator']).read_bytes()).hexdigest()==p['sha256'];m=tr.rt.load_model();h=tr.robot_height(m);paired={};checks=0;maxerr=0
for label,key in [('kick_v3','rollback'),('kick_v3_control','physical')]:
 folder=ROOT/'confirmation'/label;rows=json.loads((folder/'results.json').read_text());assert len(rows)==80
 for r in rows:
  assert not r.get('error');k=(r['source'],r['command_index']);pi=int(r['source'].split('_p')[-1].split('_')[0]);si=int(r['source'].split('_s')[-1]);assert r['seed']==53027000+pi*100+si
  if k not in paired:paired[k]={x:r[x] for x in ['task','source','command','command_index','seed']}
  else:assert all(paired[k][x]==r[x] for x in ['task','command','seed'])
  z=np.load(folder/(Path(r['path']).stem+'_uniform.npz'));states=z['qpos'];ts=z['time_s'];ref=z['reference_qpos'];want=1+np.arange(len(ref))*.05;v=tr.measure(tr.get_positions(m,ref),'kick',h);err=abs(v['quantity']-r['g1']['quantity']);assert v['event_pass']==r['g1']['event_pass'];actual=None
  if r['actual'] is not None:
   assert r['fall_time'] is None and ts[-1]>=want[-1]-1e-7;pos=tr.get_positions(m,states);pos=np.stack([np.interp(want,ts,x) for x in pos.reshape(len(pos),-1).T],1).reshape(len(ref),24,3);actual=tr.measure(pos,'kick',h);err=max(err,abs(actual['quantity']-r['actual']['quantity']));assert actual['event_pass']==r['actual']['event_pass']
  else:assert r['fall_time'] is not None or ts[-1]<want[-1]-1e-7
  assert err<1e-8;maxerr=max(maxerr,err);checks+=1;paired[k][key]=actual;paired[k][key+'_g1']=v;paired[k][key+'_human']=native_metrics(r)
rows=list(paired.values());assert len(rows)==80;summary={k:stats(rows,k,.45,.05) for k in ['rollback','physical','rollback_g1','physical_g1','rollback_human','physical_human']};out=ROOT/'final_delivery_kick_v3';out.mkdir(exist_ok=True);(out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'audited_results.json').write_text(json.dumps(rows,indent=2,default=lambda x:x.item()));(out/'audit.json').write_text(json.dumps(dict(paired_requests=80,physical_rollouts=checks,max_recalc_error=maxerr,seed_pairing_verified=True,rollback_hash_verified=True),indent=2));print(json.dumps(summary,indent=2))
