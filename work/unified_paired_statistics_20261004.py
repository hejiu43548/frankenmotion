"""Paired, task-stratified source-cluster bootstrap; never a selection score."""
import argparse,json,os,sys
from pathlib import Path
import numpy as np
R=Path('/home/pku/frankenmotion');U=R/'outputs_amass/franken_unified_20261004';B=R/'outputs_amass/franken_eleven_20261003';sys.path.insert(0,str(B/'code'));os.environ['ELEVEN_OUT']=str(B)
from audit_results import TASKS,RANGES
p=argparse.ArgumentParser();p.add_argument('--model',required=True);p.add_argument('--baseline',required=True);p.add_argument('--output',required=True);a=p.parse_args();read=lambda p:json.loads(p.read_text());key=lambda r:(r['source'],r['command_index'])
folders=[U/'evaluation'/n for n in [a.model,a.baseline]];datasets=[]
for f in folders:
 audit=read(f/'audit.json');assert audit['raw_max_error']<1e-8 and audit['overflow_warnings']==0
 rows=read(f/'audited_results.json');data={key(r):r for r in rows};assert len(data)==len(rows);datasets.append(data)
assert datasets[0].keys()==datasets[1].keys();assert read(folders[0]/'protocol.json')['split']==read(folders[1]/'protocol.json').get('split','development_validation')
for k,r in datasets[0].items():
 other=datasets[1][k];assert all(r[f]==other[f] for f in ['task','command','seed','path'])
 for field in ['human','g1']:assert abs(r[field]['quantity']-other[field]['quantity'])<1e-8 and r[field]['event_pass']==other[field]['event_pass']
rng=np.random.default_rng(8104);reps=10000;draws=np.zeros((reps,3));per_task={}
for i,task in enumerate(TASKS):
 span=RANGES[i][1]-RANGES[i][0];sources=sorted({r['source'] for r in datasets[0].values() if r['task']==task});clusters=[]
 for source in sources:
  rs=[r for r in datasets[0].values() if r['source']==source];assert len(rs)==5
  differences=[]
  for r in rs:
   values=[]
   for data in datasets:
    x=data[key(r)]['actual'];event=x is not None and x['event_pass'];values.append([min(abs(x['quantity']-r['command'])/span,1) if event else 1.,float(x is not None),float(event)])
   differences.append(np.asarray(values[0])-np.asarray(values[1]))
  clusters.append(np.mean(differences,axis=0))
 clusters=np.asarray(clusters);samples=clusters[rng.integers(0,len(clusters),size=(reps,len(clusters)))].mean(axis=1);draws+=samples/len(TASKS)
 per_task[task]=dict(sources=len(sources),mean_difference=clusters.mean(axis=0).tolist(),percentile_95=np.quantile(samples,[.025,.975],axis=0).T.tolist())
report=dict(model=a.model,baseline=a.baseline,split=read(folders[0]/'protocol.json')['split'],columns=['semantic_E_all','completion_fraction','event_fraction'],direction='model minus baseline; lower E_all is better, higher completion/event fractions are better',bootstrap_replicates=reps,seed=8104,method='Paired resampling of source clips within each task, retaining all five commands together; eleven tasks equally weighted.',macro_mean_difference=np.mean([v['mean_difference'] for v in per_task.values()],axis=0).tolist(),macro_percentile_95=np.quantile(draws,[.025,.975],axis=0).T.tolist(),per_task=per_task,limitations='Conditional on this selected model, cached templates and simulator protocol; not uncertainty across independent training runs or a causal policy-only comparison. Development intervals are descriptive after model selection.')
out=U/a.output;assert not out.exists();out.parent.mkdir(parents=True,exist_ok=True);out.write_text(json.dumps(report,indent=2));print(json.dumps({k:v for k,v in report.items() if k!='per_task'},indent=2))
