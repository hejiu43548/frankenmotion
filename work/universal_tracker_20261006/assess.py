import sys
sys.path.insert(0,"/home/pku/frankenmotion/work")
"""Raw-state audit and all-task metrics for one shared checkpoint."""
import os,sys,json,argparse,hashlib
from pathlib import Path
import numpy as np
R=Path('/home/pku/frankenmotion');U=R/'outputs_amass/universal_tracker_20261006';B=R/'outputs_amass/franken_eleven_20261003';sys.path.insert(0,str(B/'code'));os.environ['ELEVEN_OUT']=str(B)
import transfer as tr
from audit_results import TASKS,RANGES,TOLS,stats
from unified_metrics_20261004 import native_metrics
p=argparse.ArgumentParser();p.add_argument('--name',required=True);a=p.parse_args();out=U/'evaluation'/a.name;read=lambda p:json.loads(p.read_text());proto=read(out/'protocol.json');split=U/proto['split'];manifest=read(split/'manifest.json');rows=read(out/'results.json');assert len(rows)==len(manifest);ix={(r['source'],r['command_index']):r for r in rows};assert len(ix)==len(rows)
assert hashlib.sha256(Path(proto['checkpoint']).read_bytes()).hexdigest()==proto['sha256'];assert hashlib.sha256((split/'manifest.json').read_bytes()).hexdigest()==proto['manifest_sha256'];assert 'overflow' not in (out/'worker.log').read_text().lower();contract=read(out/'contract.json');assert contract['contact_capacity_nconmax']==256 and contract['constraint_capacity_njmax']==2048
assert contract.get('preview_offsets',[5,10,20] if proto['preview'] else [])==proto.get('preview_offsets',[5,10,20] if proto['preview'] else [])
m=tr.rt.load_model();height=tr.robot_height(m);result=[];maxerr=0.
for src in manifest:
 r=ix[(src['source'],src['command_index'])];assert not r.get('error'),r
 assert r['checkpoint']==proto['checkpoint'] and all(r[k]==src[k] for k in ['task','seed','command']) and r['input_path']==src['path']
 stem=Path(src['path']).stem;ref=np.load(split/'references'/(stem+'_uniform.npz'))['reference_qpos'];states=np.load(out/(stem+'_actual.npz'))['qpos'];assert np.isfinite(states).all();ts=np.arange(len(states))*.02;want=1+np.arange(len(ref))*.05;actual=None
 if r['actual'] is not None:
  assert r['termination_time'] is None and len(states)==r['target_steps'] and ts[-1]>=want[-1]-1e-7;pos=tr.get_positions(m,states);pos=np.stack([np.interp(want,ts,x) for x in pos.reshape(len(pos),-1).T],1).reshape(len(ref),24,3);actual=tr.measure(pos,src['task'],height);err=abs(actual['quantity']-r['actual']['quantity']);assert err<1e-8 and actual['event_pass']==r['actual']['event_pass'];maxerr=max(maxerr,err)
 else:assert r['termination_time'] is not None
 result.append(dict(src,human=native_metrics(src),g1=tr.measure(tr.get_positions(m,ref),src['task'],height),actual=actual))
summary={}
for i,task in enumerate(TASKS):
 rs=[r for r in result if r['task']==task];assert len(rs)==len(manifest)//11;span=RANGES[i][1]-RANGES[i][0];summary[task]={k:stats(rs,k,span,TOLS[i]) for k in ['human','g1','actual']}
 for k in ['human','g1','actual']:summary[task][k]['semantic_E_all']=float(np.mean([min(abs(r[k]['quantity']-r['command'])/span,1) if r[k] is not None and r[k]['event_pass'] else 1. for r in rs]))
aggregate=dict(requests=len(rows),complete=sum(s['actual']['measurable'] for s in summary.values()),event=sum(s['actual']['event_pass'] for s in summary.values()),joint=sum(s['actual']['joint_pass'] for s in summary.values()),macro_semantic_E_all=float(np.mean([s['actual']['semantic_E_all'] for s in summary.values()])))
(out/'audited_results.json').write_text(json.dumps(result,indent=2,default=lambda x:x.item()));(out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'audit.json').write_text(json.dumps(dict(single_checkpoint_sha256=proto['sha256'],unique_checkpoints=1,raw_max_error=maxerr,overflow_warnings=0,aggregate=aggregate),indent=2));print(json.dumps(aggregate),flush=True)
