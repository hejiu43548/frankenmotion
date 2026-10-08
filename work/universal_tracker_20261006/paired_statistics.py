"""Paired source-cluster uncertainty; commands from one source stay together.
No independence assumption over frames or five commands of the same seed.
"""
import argparse,json,sys
from pathlib import Path
import numpy as np
sys.path.insert(0,'/home/pku/frankenmotion/outputs_amass/franken_eleven_20261003/code')
from audit_results import TASKS,RANGES,TOLS
p=argparse.ArgumentParser();p.add_argument('--candidate',required=True);p.add_argument('--baseline',required=True);p.add_argument('--output',required=True);p.add_argument('--draws',type=int,default=10000);a=p.parse_args();A=json.loads(Path(a.candidate).read_text());B=json.loads(Path(a.baseline).read_text());key=lambda r:(r['task'],r['source'],r['command_index']);aa={key(r):r for r in A};bb={key(r):r for r in B};assert aa.keys()==bb.keys();assert all(aa[k]['seed']==bb[k]['seed'] and aa[k]['command']==bb[k]['command'] for k in aa);rng=np.random.default_rng(610606);bootstrap=np.zeros((a.draws,4));result={}
def values(row,tid):
 x=row['actual'];span=RANGES[tid][1]-RANGES[tid][0]
 if x is None:return np.array([0.,0.,0.,1.])
 event=bool(x['event_pass']);error=abs(x['quantity']-row['command']);return np.array([1.,float(event),float(event and error<=TOLS[tid]),min(error/span,1.) if event else 1.])
for tid,task in enumerate(TASKS):
 sources=sorted({r['source'] for r in A if r['task']==task});ca=[];cb=[];ra=[r for r in A if r['task']==task];rb=[bb[key(r)] for r in ra]
 for source in sources:
  keys=sorted(k for k in aa if k[0]==task and k[1]==source);assert len(keys)==5;ca.append(np.mean([values(aa[k],tid) for k in keys],0));cb.append(np.mean([values(bb[k],tid) for k in keys],0))
 ca=np.asarray(ca);cb=np.asarray(cb);delta=ca-cb;draw=rng.integers(len(sources),size=(a.draws,len(sources)));boot=delta[draw].mean(1);bootstrap+=boot/len(TASKS)
 entry=dict(sources=len(sources),requests=len(ra),candidate=ca.mean(0).tolist(),baseline=cb.mean(0).tolist(),difference=delta.mean(0).tolist(),paired_source_bootstrap95=np.quantile(boot,[.025,.975],axis=0).T.tolist())
 stage={}
 for name,rows in [('candidate',ra),('baseline',rb)]:
  stage[name]={}
  for field in ['human','g1','actual']:
   observed=[r for r in rows if r[field] is not None];commands=np.asarray([r['command'] for r in observed]);quantities=np.asarray([r[field]['quantity'] for r in observed]);slope=float(np.polyfit(commands,quantities,1)[0]) if len(set(commands))>1 else None;monotonic=[]
   for source in sources:
    rr=sorted([r for r in observed if r['source']==source],key=lambda r:r['command']);
    if len(rr)==5:monotonic.extend(np.diff([r[field]['quantity'] for r in rr])>=0)
   stage[name][field]=dict(observed=len(observed),slope_observed_only=slope,nondecreasing_adjacent_fraction_complete_sources=float(np.mean(monotonic)) if monotonic else None,mean_absolute_error_observed_only=float(np.mean(abs(commands-quantities))) if observed else None)
 entry['response_descriptive']=stage;result[task]=entry
common={}
for tid,task in enumerate(TASKS):
 for source in sorted({r['source'] for r in A if r['task']==task}):
  block=source.rsplit('_p',1)[1];keys=sorted(k for k in aa if k[0]==task and k[1]==source)
  common.setdefault(block,[]).append(np.mean([values(aa[k],tid)-values(bb[k],tid) for k in keys],axis=0))
assert all(len(v)==len(TASKS) for v in common.values())
block_values=np.asarray([np.mean(common[k],axis=0) for k in sorted(common)]);block_rng=np.random.default_rng(610607);block_draws=block_rng.integers(len(block_values),size=(a.draws,len(block_values)));block_bootstrap=block_values[block_draws].mean(axis=1)
assert np.max(np.abs(block_values.mean(axis=0)-np.mean([r['difference'] for r in result.values()],axis=0)))<1e-10
out=dict(metric_order=['completion_fraction','event_fraction','joint_parameter_event_fraction','semantic_E_all'],uncertainty='Paired source-cluster percentile bootstrap. Per-task draws keep each5-command source together. Primary macro common-block CI additionally keeps matching prompt/seed IDs across all11 tasks together, preserving shared generation noise. Independent-within-task macro CI retained as sensitivity. Evaluation-source uncertainty only, not training-seed uncertainty.',draws=a.draws,seed=610606,common_block_seed=610607,common_blocks=len(common),candidate=a.candidate,baseline=a.baseline,tasks=result,macro_difference=block_values.mean(axis=0).tolist(),macro_common_block_bootstrap95=np.quantile(block_bootstrap,[.025,.975],axis=0).T.tolist(),macro_paired_source_bootstrap95=np.quantile(bootstrap,[.025,.975],axis=0).T.tolist());Path(a.output).write_text(json.dumps(out,indent=2));print(json.dumps({k:out[k] for k in ['macro_difference','macro_common_block_bootstrap95','macro_paired_source_bootstrap95']},indent=2))
