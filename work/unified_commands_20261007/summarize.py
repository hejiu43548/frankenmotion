from pathlib import Path
import json,numpy as np,argparse,hashlib
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/unified_commands_20261007';TASKS=['raise_hand','reach','strike','wave','turn','sidestep','back_walk','kick','jump','lean','walk'];UNITS=['m','m','m/s','m','rad','m','m/s','m','m','rad','m/s'];out=D/'report';out.mkdir(exist_ok=True)
read=lambda p:json.loads((D/p).read_text())
b=read('generation/baseline_final/audited_results.json');s=read('generation/unified_final/audited_results.json');bp=read('general_evaluation/baseline_final/audited_results.json');sp=read('general_evaluation/unified_final/audited_results.json');ba=read('general_evaluation/baseline_final/assessment.json');sa=read('general_evaluation/unified_final/assessment.json');h=read('generation/unified_final/comparison.json');key=lambda r:(r['source'],r['command_index']);look={key(r):r for r in bp};assert len(b)==len(s)==len(bp)==len(sp)==1376
for r in sp:assert r['seed']==look[key(r)]['seed'] and r['command']==look[key(r)]['command']
summary=dict(requests=len(s),standard_requests=sum(h[t]['count'] for t in TASKS),human_before=sum(h[t]['baseline']['joint_pass'] for t in TASKS),human_after=sum(h[t]['human']['joint_pass'] for t in TASKS),physical_before=sum(ba[t]['joint_pass'] for t in TASKS),physical_after=sum(sa[t]['joint_pass'] for t in TASKS),completed_before=sum(x['complete'] for x in ba.values()),completed_after=sum(x['complete'] for x in sa.values()),physical_E_before=np.mean([ba[t]['semantic_E'] for t in TASKS]),physical_E_after=np.mean([sa[t]['semantic_E'] for t in TASKS]),human_mean_joint_difference_mm=1000*np.mean([r['joint_position_mae_m'] for r in s]),human_p95_clip_mean_joint_difference_mm=1000*np.quantile([r['joint_position_mae_m'] for r in s],.95),human_max_frame_joint_difference_mm=1000*max(r['max_joint_error_m'] for r in s),per_kind={k:dict(human=h[k],physical_before=ba[k],physical_after=sa[k]) for k in h})
# Paired episode-level bootstrap, descriptive not a universal generalization claim.
import sys
sys.path.insert(0,str(R/'outputs_amass/franken_eleven_20261003/code'))
from audit_results import TOLS
rng=np.random.default_rng(107089000);deltas=[];clusters={}
for r in sp:
 if r['kind'] not in TASKS:continue
 old=look[key(r)];tol=TOLS[TASKS.index(r['task'])]
 passed=lambda x:int(x['actual'] is not None and x['actual']['event_pass'] and abs(x['actual']['quantity']-x['command'])<=tol)
 delta=passed(r)-passed(old);deltas.append(delta);clusters.setdefault(tuple(r['source'].split('_')[-2:]),[]).append(delta)
x=np.array(deltas);cluster_means=np.array([np.mean(v) for v in clusters.values()]);boot=np.mean(cluster_means[rng.integers(0,len(cluster_means),(10000,len(cluster_means)))],axis=1);summary['paired_physical_pass_change_pp']=float(x.mean()*100);summary['paired_source_bootstrap_95ci_pp']=(np.quantile(boot,[.025,.975])*100).tolist();summary['bootstrap_limit']='Paired prompt/noise blocks resampled jointly across commands/tasks (16 blocks); descriptive only. Fixed prompt bank and frozen tracker.'
fig,axes=plt.subplots(4,3,figsize=(15,15),constrained_layout=True);colors=['#747f89','#237bb5','#55a07a','#df752f'];datasets=[b,s,bp,sp];labels=['Original human','Shared human','Original + tracker','Shared + tracker']
for tid,t in enumerate(TASKS):
 ax=axes.flat[tid]
 for data,label,color in zip(datasets,labels,colors):
  rr=[r for r in data if r['kind']==t];cs=sorted(set(r['command'] for r in rr));vv=[];lo=[];hi=[]
  for c in cs:
   qs=[(r.get('human') if 'human' in r else r['actual']) for r in rr if r['command']==c];qs=[q['quantity'] for q in qs if q is not None];vv.append(np.median(qs));lo.append(np.quantile(qs,.1));hi.append(np.quantile(qs,.9))
  ax.plot(cs,vv,'o-',label=label,color=color,ms=3);ax.fill_between(cs,lo,hi,color=color,alpha=.10)
 ax.plot(cs,cs,'k--',lw=.7,alpha=.5);ax.set_title(f'{t} | physical {ba[t]["joint_pass"]}/80 -> {sa[t]["joint_pass"]}/80');ax.set_xlabel('Command ('+UNITS[tid]+')');ax.set_ylabel('Measured ('+UNITS[tid]+')');ax.grid(alpha=.18)
axes.flat[-1].axis('off');handles,labs=axes.flat[0].get_legend_handles_labels();axes.flat[-1].legend(handles,labs,loc='upper left',frameon=False);axes.flat[-1].text(0,.52,'4 prompts x 4 seeds x 5 commands\nMedian; bands P10–P90\nSame frozen G1 tracker\nPass = event AND command tolerance\nHuman-equivalent lengths/speeds\nQ: completed rollouts only\nAll requests in pass denominator',transform=axes.flat[-1].transAxes,fontsize=10,va='top');fig.savefig(out/'command_curves.png',dpi=170);fig.savefig(out/'command_curves.pdf');plt.close(fig)
(out/'summary.json').write_text(json.dumps(summary,indent=2,default=lambda x:float(x)));print(json.dumps({k:v for k,v in summary.items() if k!='per_kind'},indent=2))
