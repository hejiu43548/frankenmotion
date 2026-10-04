"""Audit routed control against exact development references without model selection."""
import sys,os,json,hashlib,argparse
from pathlib import Path
import numpy as np
R=Path('/home/pku/frankenmotion');N=R/'outputs_amass/franken_improve_20261003';U=N.parent/'franken_unified_20261004';B=R/'outputs_amass/franken_eleven_20261003';sys.path.insert(0,str(B/'code'));os.environ['ELEVEN_OUT']=str(B)
import transfer as tr
from audit_results import TASKS,RANGES,TOLS,stats
from unified_metrics_20261004 import native_metrics
p=argparse.ArgumentParser();p.add_argument('--name',default='routed_baseline_validation');a=p.parse_args();read=lambda p:json.loads(p.read_text());out=U/'evaluation'/a.name;proto=read(out/'protocol.json');split=U/proto.get('split','development_validation');manifest=read(split/'manifest.json');sonic=read(out/'sonic/results.json');bm=read(out/'bm/results.json');assert len(sonic)+len(bm)==len(manifest) and len(sonic)*3==len(bm)*8;ix=lambda rows:{(r['source'],r['command_index']):r for r in rows};ss=ix(sonic);bb=ix(bm);assert len(ss)==len(sonic) and len(bb)==len(bm)
assert hashlib.sha256((split/'manifest.json').read_bytes()).hexdigest()==proto['manifest_sha256'];assert 'overflow' not in (out/'bm.log').read_text().lower();wm=read(N/'frozen_controllers_v1/weight_map.json');contract=read(out/'bm/contract.json');assert contract['contact_capacity_nconmax']==256 and contract['constraint_capacity_njmax']==2048
m=tr.rt.load_model();height=tr.robot_height(m);result=[];maxerr=0.
for src in manifest:
 key=(src['source'],src['command_index']);route=proto['task_route'][src['task']];r=ss[key] if route=='sonic' else bb[key];assert not r.get('error');assert all(r[k]==src[k] for k in ['task','seed','command']);stem=Path(src['path']).stem;ref=np.load(split/'references'/(stem+'_uniform.npz'))['reference_qpos'];want=1+np.arange(len(ref))*.05
 if route=='sonic':
  assert r['path']==src['path'];z=np.load(out/'sonic'/(stem+'_uniform.npz'));states=z['qpos'];ts=z['time_s'];fall=r['fall_time'];assert np.allclose(ts,(np.arange(len(states))+1)*.02)
 else:
  assert r['input_path']==src['path'] and r['checkpoint']==wm[src['task']];states=np.load(out/'bm'/(stem+'_actual.npz'))['qpos'];ts=np.arange(len(states))*.02;fall=r['termination_time']
 assert np.isfinite(states).all();actual=None
 if r['actual'] is not None:
  assert fall is None and ts[-1]>=want[-1]-1e-7;pos=tr.get_positions(m,states);pos=np.stack([np.interp(want,ts,x) for x in pos.reshape(len(pos),-1).T],1).reshape(len(ref),24,3);actual=tr.measure(pos,src['task'],height);err=abs(actual['quantity']-r['actual']['quantity']);assert err<1e-8 and actual['event_pass']==r['actual']['event_pass'];maxerr=max(maxerr,err)
 else:assert fall is not None or ts[-1]<want[-1]-1e-7
 result.append(dict(src,controller=route,human=native_metrics(src),g1=tr.measure(tr.get_positions(m,ref),src['task'],height),actual=actual))
summary={}
for i,task in enumerate(TASKS):
 rs=[r for r in result if r['task']==task];assert len(rs)==len(manifest)//11;span=RANGES[i][1]-RANGES[i][0];summary[task]={k:stats(rs,k,span,TOLS[i]) for k in ['human','g1','actual']}
 for k in ['human','g1','actual']:summary[task][k]['semantic_E_all']=float(np.mean([min(abs(r[k]['quantity']-r['command'])/span,1) if r[k] is not None and r[k]['event_pass'] else 1 for r in rs]))
aggregate=dict(requests=len(manifest),complete=sum(s['actual']['measurable'] for s in summary.values()),event=sum(s['actual']['event_pass'] for s in summary.values()),joint=sum(s['actual']['joint_pass'] for s in summary.values()),macro_semantic_E_all=float(np.mean([s['actual']['semantic_E_all'] for s in summary.values()])))
(out/'audited_results.json').write_text(json.dumps(result,indent=2,default=lambda x:x.item()));(out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'audit.json').write_text(json.dumps(dict(scope='routed baseline, not unified',raw_max_error=maxerr,overflow_warnings=0,aggregate=aggregate),indent=2));print(json.dumps(aggregate))
