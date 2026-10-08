import json, numpy as np
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import D,TASKS,RANGES
TOLS=[.02,.02,.1,.015,.08,.06,.05,.05,.04,.06,.05]
out=D/'report';out.mkdir(exist_ok=True)
names=['teacher_final','unified_final'];data={n:json.loads((D/'general_evaluation'/n/'audited_results.json').read_text()) for n in names};summary={n:json.loads((D/'general_evaluation'/n/'eleven_summary.json').read_text()) for n in names}
fig,axes=plt.subplots(4,3,figsize=(15,15));colors=['#708090','#208ac0','#df762e','#279577'];series=[('teacher_final','human','Old routed human'),('unified_final','human','Unified human'),('teacher_final','actual','Old + same tracker'),('unified_final','actual','Unified + same tracker')]
for tid,task in enumerate(TASKS):
 ax=axes.flat[tid]
 for color,(name,stage,label) in zip(colors,series):
  rr=[r for r in data[name] if r['task']==task];cmds=sorted(set(r['command'] for r in rr));means=[];low=[];high=[]
  for cmd in cmds:
   vals=[r[stage]['quantity'] for r in rr if r['command']==cmd and r[stage] is not None];means.append(np.mean(vals) if vals else np.nan);low.append(np.quantile(vals,.1) if vals else np.nan);high.append(np.quantile(vals,.9) if vals else np.nan)
  ax.plot(cmds,means,'o-',color=color,label=label,linewidth=1.5,markersize=3);ax.fill_between(cmds,low,high,color=color,alpha=.08)
 ax.plot(RANGES[tid],RANGES[tid],'--',color='gray',linewidth=1);ax.set_title(task+' | physics '+str(summary['unified_final'][task]['actual']['joint_pass'])+'/80 pass');unit=['m','m','m/s','m','rad','m','m/s','m','m','rad','m/s'][tid];ax.set_xlabel('Command ('+unit+')');ax.set_ylabel('Human-equivalent Q ('+unit+')');ax.grid(alpha=.15)
axes.flat[11].axis('off');handles,labels=axes.flat[0].get_legend_handles_labels();axes.flat[11].legend(handles,labels,loc='upper left');axes.flat[11].text(0,.45,'11 tasks; 4 fixed prompts x 4 fresh noises x 5 commands\n\nGray dashed: ideal response; bands: P10-P90\nIncomplete physics omitted from Q\nPass count includes every request, no failure exclusion\nSame frozen tracker + same retarget protocol\nNo task-dependent generator weight switching',fontsize=10,va='top');fig.tight_layout();fig.savefig(out/'command_response.png',dpi=180);fig.savefig(out/'command_response.pdf');plt.close(fig)
rng=np.random.default_rng(107079);ci={}
for stage in ['human','actual']:
 clusters=[];rows=[]
 for tid,task in enumerate(TASKS):
  aa={r['source']+'_'+str(r['command_index']):r for r in data[names[0]] if r['task']==task};bb={r['source']+'_'+str(r['command_index']):r for r in data[names[1]] if r['task']==task};assert aa.keys()==bb.keys();by={}
  for key,a in aa.items():
   b=bb[key];cmd=a['command'];span=RANGES[tid][1]-RANGES[tid][0]
   def value(r):
    v=r[stage];ok=v is not None and v['event_pass'];return [float(ok and abs(v['quantity']-cmd)<=TOLS[tid]),min(abs(v['quantity']-cmd)/span,1) if ok else 1.]
   by.setdefault(a['source'],[]).append(np.array(value(b))-np.array(value(a)))
  cluster=np.array([np.mean(v,axis=0) for _,v in sorted(by.items())]);clusters.append(cluster)
  rows.append(dict(task=task,old=summary[names[0]][task][stage],unified=summary[names[1]][task][stage]))
 idx=rng.integers(0,16,size=(10000,16));draws=np.mean([v[idx].mean(1) for v in clusters],axis=0);delta=np.mean([v.mean(0) for v in clusters],axis=0);ci[stage]=dict(pass_rate_difference=float(delta[0]),pass_rate_difference_ci95=np.quantile(draws[:,0],[.025,.975]).tolist(),semantic_E_difference=float(delta[1]),semantic_E_difference_ci95=np.quantile(draws[:,1],[.025,.975]).tolist(),per_task=rows)
ci['interpretation']='Paired bootstrap resamples the same 16 prompt-index/noise-index blocks jointly across all11 fixed tasks, preserving all5 commands within each block and paired generator outputs. Conditional on fixed prompt templates and nominal simulator. Not a test of generalization to arbitrary language, unseen tasks, or real hardware.'
(out/'comparison.json').write_text(json.dumps(ci,indent=2))
manifest=json.loads((D/'generation/unified_final/manifest.json').read_text());assert len(manifest)==880;assert len(set(r['generator_sha256'] for r in manifest))==1
(out/'one_checkpoint_audit.json').write_text(json.dumps(dict(requests=880,unique_generator_hashes=list(set(r['generator_sha256'] for r in manifest)),tasks=sorted(set(r['task'] for r in manifest)),distinct_commands_per_task={t:len(set(r['command'] for r in manifest if r['task']==t)) for t in TASKS}),indent=2))
print(json.dumps({k:{x:v for x,v in c.items() if x!='per_task'} for k,c in ci.items() if isinstance(c,dict)},indent=2))
