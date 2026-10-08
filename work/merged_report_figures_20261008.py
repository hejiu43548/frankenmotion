import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path('/home/pku/frankenmotion/outputs_amass/merged_report_20261008');out=D/'partner_style_report';out.mkdir(exist_ok=True)
r=json.loads((D/'general_evaluation/unified_final/audited_results.json').read_text());o=json.loads((D/'general_evaluation/baseline_final/audited_results.json').read_text());s=json.loads((D/'general_evaluation/unified_final/eleven_summary.json').read_text());os=json.loads((D/'general_evaluation/baseline_final/eleven_summary.json').read_text());tasks=list(s);tols=[.02,.02,.1,.015,.08,.06,.05,.05,.04,.06,.05];units=['m','m','m/s','m','rad','m','m/s','m','m','rad','m/s'];titles=['Raise hand','Forward reach','Strike speed','Wave amplitude','Right turn','Right sidestep','Backward walk','Right kick','Jump height','Held forward lean','Forward walk'];colors=['#8495a1','#0086b6','#ef7d36','#499e7d'];rng=np.random.default_rng(71107);meta=[]
plt.rcParams.update({'font.size':10,'axes.grid':True,'grid.alpha':.15,'axes.titleweight':'bold'})
def response(ax,rr,oo,cs,unit,title,small=False):
 for rows,key,label,color,ls in [(rr,'human','Human reference',colors[0],'--'),(rr,'g1','G1 converted reference',colors[1],'-'),(rr,'actual','Current tracker execution',colors[2],'-'),(oo,'actual','Before merge + same tracker',colors[3],'--')]:
  vals=[[x[key]['quantity'] for x in rows if x['command']==c and x[key] is not None] for c in cs];m=[np.mean(x) for x in vals];ax.plot(cs,m,'o'+ls,color=color,label=label,lw=1.6,ms=3);ax.fill_between(cs,[np.quantile(x,.1) for x in vals],[np.quantile(x,.9) for x in vals],color=color,alpha=.09)
  if key=='actual' and rows is rr:
   for c,q,v in zip(cs,m,vals):ax.annotate(f'{len(v)}/16',(c,q),xytext=(0,7),textcoords='offset points',fontsize=7,color=color,ha='center')
 ax.plot(cs,cs,'--',color='#8190a3',lw=1,label='Command target');ax.set_title(title,loc='left',fontsize=10 if small else 12);ax.set_xlabel('Human command ('+unit+')');ax.set_ylabel('Human-equivalent Q ('+unit+')')
figall,axall=plt.subplots(4,3,figsize=(15,9.4))
for i,t in enumerate(tasks):
 rr=sorted([x for x in r if x['task']==t],key=lambda x:(x['source'],x['command_index']));oo=sorted([x for x in o if x['task']==t],key=lambda x:(x['source'],x['command_index']));assert [(x['source'],x['command']) for x in rr]==[(x['source'],x['command']) for x in oo];cs=sorted({x['command'] for x in rr});span=cs[-1]-cs[0]
 response(axall.flat[i],rr,oo,cs,units[i],titles[i]+f' | {s[t]["actual"]["measurable"]}/80 measurable',True)
 fig,axs=plt.subplots(3,1,figsize=(10.1,8.3),gridspec_kw={'height_ratios':[3,1.35,1.1]});response(axs[0],rr,oo,cs,units[i],titles[i]+f' | {s[t]["actual"]["measurable"]}/80 measurable');axs[0].legend(fontsize=7,ncol=2,loc='upper left')
 for j,(label,func) in enumerate([('Generation',lambda x:x['human']['quantity']-x['command']),('Conversion',lambda x:x['g1']['quantity']-x['human']['quantity']),('Tracking',lambda x:x['actual']['quantity']-x['g1']['quantity']),('Final',lambda x:x['actual']['quantity']-x['command'])]):
  vals=[[func(x)/span for x in rr if x['command']==c and (j<2 or x['actual'] is not None)] for c in cs];col=(colors+['#21364e'])[j] if j<3 else '#21364e';axs[1].plot(cs,[np.mean(v) for v in vals],'o-',label=label,color=col,ms=3);axs[1].fill_between(cs,[np.quantile(v,.1) for v in vals],[np.quantile(v,.9) for v in vals],color=col,alpha=.08)
 axs[1].axhline(0,color='gray',lw=.5);axs[1].set_title('Signed error layers / command span (P10-P90)',loc='left',fontsize=10);axs[1].set_ylabel('Normalized error');axs[1].legend(ncol=4,fontsize=7,loc='upper right')
 width=(cs[1]-cs[0])*.16
 checks=[('Measured',lambda x:x['actual'] is not None),('Complete',lambda x:x['physical_complete']),('Event',lambda x:x['actual'] is not None and x['actual']['event_pass']),('Joint',lambda x:x['actual'] is not None and x['actual']['event_pass'] and abs(x['actual']['quantity']-x['command'])<=tols[i])]
 for j,(name,check) in enumerate(checks):axs[2].bar(np.array(cs)+(j-1.5)*width,[sum(check(x) for x in rr if x['command']==c) for c in cs],width,label=name,color=['#e99866','#5d9db4','#7ea58d','#9682b5'][j])
 axs[2].set_ylim(0,18);axs[2].set_xticks(cs);axs[2].set_ylabel('Count / 16');axs[2].set_title('Coverage by command (all planned requests retained)',loc='left',fontsize=10);axs[2].legend(ncol=4,fontsize=7,loc='upper right');fig.tight_layout();fig.savefig(out/(t+'.png'),dpi=180);plt.close(fig)
 def err(x):return min(abs(x['actual']['quantity']-x['command'])/span,1) if x['actual'] is not None else 1
 diffs=np.array([err(x)-err(y) for x,y in zip(rr,oo)]).reshape(16,5).mean(1);boot=diffs[rng.integers(0,16,(10000,16))].mean(1)
 meta.append(dict(task=t,title=titles[i],unit=units[i],tol=tols[i],stats=s[t],old=os[t],delta=float(diffs.mean()),ci=np.quantile(boot,[.025,.975]).tolist(),conversion_mae_span=float(np.mean([abs(x['g1']['quantity']-x['human']['quantity'])/span for x in rr])),paired=sum(x['actual'] is not None and y['actual'] is not None for x,y in zip(rr,oo)),complete=sum(x['physical_complete'] for x in rr)))
axall.flat[11].axis('off');h,l=axall.flat[0].get_legend_handles_labels();axall.flat[11].legend(h,l,loc='upper left',fontsize=9);axall.flat[11].text(0,.40,'11 tasks | 16 sources x 5 commands\nP10-P90 bands; labels: measured / 16\nMissing Q omitted from curves, scored 1 in E_all\nFixed prompts; fresh noise; same frozen tracker',fontsize=9,va='top');figall.tight_layout();figall.savefig(out/'overview.png',dpi=190);plt.close(figall);(out/'metrics.json').write_text(json.dumps(meta,indent=2))
