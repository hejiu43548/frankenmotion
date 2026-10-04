"""Recompute all reported quantities from archived physical states after freeze."""
import sys,json
from pathlib import Path
import numpy as np
BASE=Path('/home/pku/frankenmotion/outputs_amass/franken_eleven_20261003');ROOT=BASE.parent/'franken_improve_20261003';sys.path.insert(0,str(BASE/'code'));import transfer as tr
assert (ROOT/'frozen_controllers_v1/protocol.json').exists(),'Freeze selection first'
m=tr.rt.load_model();height=tr.robot_height(m);checks=[];largest=0.
for folder,expected in [(ROOT/'confirmation/paired_baseline',1760),(ROOT/'confirmation/candidate',880)]:
 rows=json.loads((folder/'results.json').read_text());assert len(rows)==expected
 for r in rows:
  if r.get('error'):checks.append(dict(source=r['source'],method=r['method'],status='recorded_error',error=r['error']));continue
  stem=Path(r['path']).stem;z=np.load(folder/(stem+'_'+r['method']+'.npz'));states=z['qpos'];ts=z['time_s'];ref=z['reference_qpos'];assert len(states)==len(ts) and np.isfinite(states).all();assert np.allclose(ts,(np.arange(len(ts))+1)*.02,atol=1e-10);want=1+np.arange(len(ref))*.05
  rg=tr.measure(tr.get_positions(m,ref),r['task'],height);err=abs(rg['quantity']-r['g1']['quantity']);assert err<1e-8
  if r['actual'] is not None:
   assert r['fall_time'] is None and ts[-1]>=want[-1]-1e-7
   p=tr.get_positions(m,states);p=np.stack([np.interp(want,ts,x) for x in p.reshape(len(p),-1).T],1).reshape(len(ref),24,3);v=tr.measure(p,r['task'],height);err=max(err,abs(v['quantity']-r['actual']['quantity']));assert v['event_pass']==r['actual']['event_pass'];assert err<1e-8
  else:assert r['fall_time'] is not None or ts[-1]<want[-1]-1e-7
  largest=max(largest,err);checks.append(dict(source=r['source'],command_index=r['command_index'],method=r['method'],status='verified',quantity_recalc_error=err))
folder=ROOT/'beyondmimic_confirmation';rows=json.loads((folder/'results.json').read_text());assert len(rows)==880
for r in rows:
 if r.get('error'):checks.append(dict(source=r['source'],method='beyondmimic',status='recorded_error',error=r['error']));continue
 assert r['termination_criterion']=='physical' and r['initialization']=='standing + 1s entry';stem=Path(r['input_path']).stem;states=np.load(folder/(stem+'_actual.npz'))['qpos'];assert np.isfinite(states).all()
 if r['actual'] is not None:
  assert r['termination_time'] is None and len(states)==r['target_steps'];n=120 if r['task'] in ['wave','turn','sidestep','back_walk','walk'] else 60;want=1+np.arange(n)*.05;ts=np.arange(len(states))*.02;assert ts[-1]>=want[-1];p=tr.get_positions(m,states);p=np.stack([np.interp(want,ts,x) for x in p.reshape(len(p),-1).T],1).reshape(n,24,3);v=tr.measure(p,r['task'],height);err=abs(v['quantity']-r['actual']['quantity']);assert err<1e-8 and v['event_pass']==r['actual']['event_pass'];largest=max(largest,err)
 else:assert r['termination_time'] is not None
 checks.append(dict(source=r['source'],command=r['command'],method='beyondmimic',status='verified'))
report=dict(planned_rollouts=3520,records_checked=len(checks),raw_recalculation_max_abs_error=largest,recorded_errors=sum(r['status']=='recorded_error' for r in checks),note='Errors retained in fixed planned denominator. Physical-state trajectories, not reference animation, were remeasured.',checks=checks);out=ROOT/'final_delivery';out.mkdir(exist_ok=True);(out/'trajectory_audit.json').write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='checks'}))
