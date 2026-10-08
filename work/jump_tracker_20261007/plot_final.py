"""Final locked-test plots. Failed requests stay in pass-rate denominators."""
import argparse,json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--root',required=True);a=p.parse_args();D=Path(a.root);out=D/'deliverables';out.mkdir(exist_ok=True)
old=json.load(open(D/'general_evaluation/final_baseline/audited_results.json'));new=json.load(open(D/'general_evaluation/final_candidate/audited_results.json'))
def key(r):return (r['source'],r.get('seed'),round(r['command'],6))
ob={key(r):r for r in old};nb={key(r):r for r in new};assert ob.keys()==nb.keys() and len(ob)==60
def passed(r):return bool(r.get('physical_complete') and r.get('actual') and r['actual']['event_pass'] and abs(r['actual']['quantity']-r['command'])<=.04)
def summarize(rows):
 measured=[r for r in rows if r.get('actual')];errors=[abs(r['actual']['quantity']-r['command']) for r in measured]
 return dict(requests=len(rows),complete=sum(r.get('physical_complete',False) for r in rows),event=sum(bool(r.get('actual') and r['actual']['event_pass']) for r in rows),height_and_event_pass=sum(passed(r) for r in rows),mae_completed_m=float(np.mean(errors)) if errors else None,semantic_E_all=float(np.mean([min(abs(r['actual']['quantity']-r['command'])/.3,1.) if r.get('actual') and r['actual']['event_pass'] else 1. for r in rows])))
summary={};records=[]
for part in ['all','standard_grid','interpolation']:
 rr=[r for r in new if part=='all' or r['command_set']==part];bb=[ob[key(r)] for r in rr];summary[part]={'baseline':summarize(bb),'candidate':summarize(rr)}
for r in new:
 b=ob[key(r)];records.append(dict(source=r['source'],seed=r.get('seed'),command_set=r['command_set'],command=r['command'],g1_reference_height=r['g1']['quantity'],baseline_height=b['actual']['quantity'] if b.get('actual') else None,candidate_height=r['actual']['quantity'] if r.get('actual') else None,baseline_complete=b['physical_complete'],candidate_complete=r['physical_complete'],baseline_pass=passed(b),candidate_pass=passed(r)))
with (out/'paired_fresh_results.csv').open('w',newline='') as f:
 w=csv.DictWriter(f,fieldnames=records[0]);w.writeheader();w.writerows(records)
# Eight independently generated prompt/noise source groups, commands paired within group.
rng=np.random.default_rng(710799);groups=sorted({(r['source'],r.get('seed')) for r in new});diffs=[]
for _ in range(10000):
 drawn=rng.integers(len(groups),size=len(groups));rr=[r for i in drawn for r in new if (r['source'],r.get('seed'))==groups[i]];diffs.append(np.mean([int(passed(r))-int(passed(ob[key(r)])) for r in rr]))
summary['paired_cluster_bootstrap']=dict(groups=len(groups),resamples=10000,seed=710799,pass_rate_gain_95_percentile_interval=np.percentile(diffs,[2.5,97.5]).tolist(),scope='Exploratory source-cluster bootstrap;8 groups with repeated prompt templates. Not unseen-language or hardware evidence.')
(out/'final_summary.json').write_text(json.dumps(summary,indent=2))
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
fig,axes=plt.subplots(2,2,figsize=(12.5,9.2),gridspec_kw={'height_ratios':[1.35,1]},layout='constrained')
for col,part in enumerate(['standard_grid','interpolation']):
 rr=[r for r in new if r['command_set']==part];bb=[ob[key(r)] for r in rr];xs=sorted({r['command'] for r in rr});ax=axes[0,col]
 ax.plot([.23,.57],[.23,.57],color='#bdc5ce',ls=':',label='Requested height')
 for rows,field,label,color,style in [(rr,'g1','G1 reference','#778694','--'),(bb,'actual','Original tracker','#dd7237','-'),(rr,'actual','New shared tracker','#1675b8','-')]:
  arrays=[np.array([r[field]['quantity'] for r in rows if abs(r['command']-x)<1e-8 and r.get(field)]) for x in xs];means=[z.mean() if len(z) else np.nan for z in arrays];lo=[np.percentile(z,10) if len(z) else np.nan for z in arrays];hi=[np.percentile(z,90) if len(z) else np.nan for z in arrays];ax.plot(xs,means,style+'o',color=color,label=label,lw=2,ms=4);ax.fill_between(xs,lo,hi,color=color,alpha=.10)
 ss=summary[part];ax.set(title=('Standard command grid' if col==0 else 'Intermediate command values')+f' | n={len(rr)}',xlabel='Requested height (human-equivalent m)',ylabel='Measured height (human-equivalent m)',xlim=(.23,.57),ylim=(.04,.61));ax.grid(alpha=.15);ax.legend(fontsize=9,loc='upper left')
 ax=axes[1,col];width=.018 if col==0 else .010
 for rows,label,color,off in [(bb,'Original','#dd7237',-width/2),(rr,'New shared','#1675b8',width/2)]:
  grouped=[[r for r in rows if abs(r['command']-x)<1e-8] for x in xs];counts=[sum(passed(r) for r in g) for g in grouped];rates=[v/len(g)*100 for v,g in zip(counts,grouped)];ax.bar(np.array(xs)+off,rates,width*.9,color=color,label=label)
  for x,y,k,g in zip(xs,rates,counts,grouped):ax.text(x+off,y+3,f'{k}/{len(g)}',ha='center',fontsize=8,color=color)
 ax.set(xlabel='Requested height (human-equivalent m)',ylabel='Complete + event + height within 4 cm (%)',ylim=(0,115),xticks=xs,title=f'Overall pass: {ss["baseline"]["height_and_event_pass"]}/{len(rr)} -> {ss["candidate"]["height_and_event_pass"]}/{len(rr)}; complete: {ss["baseline"]["complete"]} -> {ss["candidate"]["complete"]}');ax.grid(axis='y',alpha=.15)
 if col==1:ax.tick_params(axis='x',labelrotation=20)
fig.suptitle('FrankenMotion -> GMR -> G1: frozen tracker comparison on fresh diffusion noise',fontsize=15)
fig.supxlabel('Actual curves: completed rollouts only, mean and P10-P90. Pass bars retain every request, including falls.\nGenerator, retargeting, robot limits and references are identical. Human-equivalent scaling = 1.270119 / 1.048644.',fontsize=10)
fig.savefig(out/'fresh_jump_comparison.png',dpi=180);fig.savefig(out/'fresh_jump_comparison.svg');plt.close(fig)
print(json.dumps(summary,indent=2))
