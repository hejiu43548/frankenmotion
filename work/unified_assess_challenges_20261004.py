"""Audit full continuous trajectories, including resets and component measurements."""
import sys,json,hashlib,argparse
from pathlib import Path
import numpy as np
R=Path('/home/pku/frankenmotion');U=R/'outputs_amass/franken_unified_20261004';sys.path.insert(0,str(R/'work'))
from unified_challenges_20261004 import sequence_metrics,tr
from audit_results import TASKS,TOLS
p=argparse.ArgumentParser();p.add_argument('--name',required=True);a=p.parse_args();out=U/'evaluation'/a.name;read=lambda p:json.loads(p.read_text());proto=read(out/'protocol.json');source=Path(proto['source']);manifest=read(source/'manifest.json');rows=read(out/'results.json');assert len(manifest)==len(rows)==36;ix={r['source']:r for r in rows};assert len(ix)==36;assert hashlib.sha256(Path(proto['checkpoint']).read_bytes()).hexdigest()==proto['sha256'];assert hashlib.sha256((source/'manifest.json').read_bytes()).hexdigest()==proto['manifest_sha256'];assert 'overflow' not in (out/'worker.log').read_text().lower();m=tr.rt.load_model();maxerr=0.;result=[]
for src in manifest:
 r=ix[src['source']];assert not r.get('error'),r;assert r['checkpoint']==proto['checkpoint'] and r['input_path']==src['path'] and r['seed']==src['seed'];assert r['checkpoint_loads']==1
 stem=Path(src['path']).stem;ref=np.load(source/'references'/(stem+'_uniform.npz'))['reference_qpos'];states=np.load(out/(stem+'_actual.npz'))['qpos'];ts=np.arange(len(states))*.02;want=1+np.arange(len(ref))*.05;assert np.isfinite(states).all();actual=None
 if r['actual'] is not None:
  assert r['internal_reset_count']==0 and r['termination_time'] is None and len(states)==r['target_steps'] and ts[-1]>=want[-1]-1e-7;pos=tr.get_positions(m,states);pos=np.stack([np.interp(want,ts,x) for x in pos.reshape(len(pos),-1).T],1).reshape(len(ref),24,3);actual=sequence_metrics(pos,ref,src,m,tr);err=abs(actual['root_position_rmse_m']-r['actual']['root_position_rmse_m']);maxerr=max(maxerr,err);assert err<1e-8
  for c,old in zip(actual['components'],r['actual']['components']):
   assert c['task']==old['task'];err=abs(c['actual']['quantity']-old['actual']['quantity']);maxerr=max(maxerr,err);assert err<1e-8 and c['actual']['event_pass']==old['actual']['event_pass'];c['joint_pass']=bool(c['actual']['event_pass'] and abs(c['actual']['quantity']-c['command'])<=TOLS[TASKS.index(c['task'])])
  actual['all_component_joint_pass']=all(c['joint_pass'] for c in actual['components'])
 else:assert r['termination_time'] is not None and r['internal_reset_count']==1
 result.append(dict(src,actual=actual,termination_time=r['termination_time'],internal_reset_count=r['internal_reset_count']))
summary={}
for kind in ['sequence','composition']:
 rs=[r for r in result if r['task']==kind];summary[kind]=dict(planned=len(rs),complete=sum(r['actual'] is not None for r in rs),all_component_events=sum(r['actual'] is not None and r['actual']['all_component_events'] for r in rs),all_component_joint_pass=sum(r['actual'] is not None and r['actual']['all_component_joint_pass'] for r in rs))
(out/'audited_results.json').write_text(json.dumps(result,indent=2,default=lambda x:x.item()));(out/'audit.json').write_text(json.dumps(dict(raw_max_error=maxerr,checkpoint_sha256=proto['sha256'],unique_checkpoints=1,overflow_warnings=0,no_internal_resets_on_completed_rollouts=True,summary=summary),indent=2));print(json.dumps(summary),flush=True)
