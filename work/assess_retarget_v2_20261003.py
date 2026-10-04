"""Recompute and report separate v2 heldout-noise results; never replace v1."""
import os,sys,json,hashlib,csv
from pathlib import Path
import numpy as np
BASE=Path('/home/pku/frankenmotion/outputs_amass/franken_eleven_20261003');ROOT=BASE.parent/'franken_improve_20261003';sys.path.insert(0,str(BASE/'code'));os.environ['ELEVEN_OUT']=str(BASE)
from audit_results import native_metrics,stats
import transfer as tr
protocol=json.loads((ROOT/'frozen_retarget_v2/protocol.json').read_text());assert hashlib.sha256((ROOT/'frozen_retarget_v2/task_retarget_v2_20261003.py').read_bytes()).hexdigest()==protocol['script_sha256']
folder=ROOT/'task_retarget_v2_confirmation';rows=json.loads((folder/'results.json').read_text());assert len(rows)==320
manifest=json.loads((ROOT/'retarget_v2_confirmation_manifest.json').read_text());assert len(manifest)==160
index={(r['source'],r['command_index'],r['weight']):r for r in rows};assert len(index)==320;m=tr.rt.load_model();height=tr.robot_height(m);maxerr=0.;result=[]
for src in manifest:
 pi=int(src['source'].split('_p')[-1].split('_')[0]);si=int(src['source'].split('_s')[-1]);assert src['seed']==53027000+pi*100+si
 out=dict(src,human=native_metrics(src))
 for w in [0,4]:
  r=index[(src['source'],src['command_index'],w)];assert all(r[x]==src[x] for x in ['path','command','seed','task']);assert not r.get('error'),r.get('error');z=np.load(folder/(Path(r['path']).stem+f'_w{w}.npz'));states=z['qpos'];ts=z['time_s'];ref=z['reference_qpos'];want=1+np.arange(len(ref))*.05;assert np.isfinite(states).all() and len(states)==len(ts)
  v=tr.measure(tr.get_positions(m,ref),r['task'],height);err=abs(v['quantity']-r['g1']['quantity']);assert v['event_pass']==r['g1']['event_pass'];out[f'g1_w{w}']=v
  actual=None
  if r['actual'] is not None:
   assert r['fall_time'] is None and ts[-1]>=want[-1]-1e-7;p=tr.get_positions(m,states);p=np.stack([np.interp(want,ts,x) for x in p.reshape(len(p),-1).T],1).reshape(len(ref),24,3);actual=tr.measure(p,r['task'],height);err=max(err,abs(actual['quantity']-r['actual']['quantity']));assert actual['event_pass']==r['actual']['event_pass']
  else:assert r['fall_time'] is not None or ts[-1]<want[-1]-1e-7
  assert err<1e-8;maxerr=max(maxerr,err);out[f'actual_w{w}']=actual
 result.append(out)
summary={};rng=np.random.default_rng(53103)
for task,span,tol in [('raise_hand',.4,.02),('lean',.5,.06)]:
 a=[r for r in result if r['task']==task];summary[task]={k:stats(a,k,span,tol) for k in ['human','g1_w0','actual_w0','g1_w4','actual_w4']};d=[]
 for source in sorted({r['source'] for r in a}):
  group=[r for r in a if r['source']==source];err=lambda k:np.mean([min(abs(r[k]['quantity']-r['command'])/span,1) if r[k] is not None else 1 for r in group]);d.append(err('actual_w4')-err('actual_w0'))
 b=np.mean(np.array(d)[rng.integers(0,16,(5000,16))],axis=1);summary[task]['paired_delta_E_all']=dict(mean=float(np.mean(d)),source_bootstrap_95=np.quantile(b,[.025,.975]).tolist())
out=ROOT/'final_delivery_v2';out.mkdir(exist_ok=True);(out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'audited_results.json').write_text(json.dumps(result,indent=2,default=lambda x:x.item()));(out/'audit.json').write_text(json.dumps(dict(requests=160,physical_rollouts=320,raw_recalc_max_error=maxerr,frozen_script_hash_verified=True,seed_pairing_verified=True,scope='Two-task separate post-v1 validation; never merge with v1 as a common-seed benchmark'),indent=2))
flat=[]
for r in result:
 q={k:r[k] for k in ['task','source','seed','command','command_index']}
 for k in ['human','g1_w0','actual_w0','g1_w4','actual_w4']:q[k+'_Q']=r[k]['quantity'] if r[k] else None;q[k+'_event']=r[k]['event_pass'] if r[k] else None
 flat.append(q)
with (out/'requests.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=flat[0]);w.writeheader();w.writerows(flat)
print(json.dumps(summary,indent=2))
