"""Publication-style paired command curves. Missing physical quantities stay blank."""
from pathlib import Path
import json,argparse
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--candidate',type=Path,required=True);p.add_argument('--baseline',type=Path,required=True);p.add_argument('--summary',type=Path,required=True);p.add_argument('--sonic',type=Path);p.add_argument('--output',type=Path,required=True);p.add_argument('--scope',default='Development only');p.add_argument('--tasks',nargs='+');a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
x=json.loads(a.candidate.read_text());b=json.loads(a.baseline.read_text());summary=json.loads(a.summary.read_text());sonic=json.loads(a.sonic.read_text()) if a.sonic else None
assert {(r['task'],r['source'],r['command_index']) for r in x}=={(r['task'],r['source'],r['command_index']) for r in b}
if sonic:assert {(r['task'],r['source'],r['command_index']) for r in x}=={(r['task'],r['source'],r['command_index']) for r in sonic}
tasks=['raise_hand','reach','strike','wave','turn','sidestep','back_walk','kick','jump','lean','walk'];titles=['Raise hand','Forward reach','Strike speed','Wave amplitude','Right turn','Right sidestep','Backward walk','Right kick','Jump height','Held forward lean','Forward walk'];units=['m','m','m/s','m','rad','m','m/s','m','m','rad','m/s']
if a.tasks:
 indices=[tasks.index(t) for t in a.tasks];tasks=[tasks[i] for i in indices];titles=[titles[i] for i in indices];units=[units[i] for i in indices]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'axes.grid':True,'grid.alpha':.18,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,axes=plt.subplots(int(np.ceil(len(tasks)/2)) if a.tasks else 4,2 if a.tasks else 3,figsize=(10.5,3.5*int(np.ceil(len(tasks)/2))+1.0) if a.tasks else (13.8,14.6),squeeze=False);colors={'human':'#7C8797','g1':'#1D8FB9','baseline':'#876CA9','sonic':'#50966D','actual':'#D56A35'};handles=[]
for ax,task,title,unit in zip(axes.flat,tasks,titles,units):
 rr=[r for r in x if r['task']==task];bb=[r for r in b if r['task']==task];commands=sorted({r['command'] for r in rr});ax.plot(commands,commands,color='#8D939A',lw=1,ls=':',zorder=0)
 curves=[('human',rr,'human','FrankenMotion human reference'),('g1',rr,'g1','G1 converted reference'),('baseline',bb,'actual','Previous shared tracker')]
 if sonic:curves.append(('sonic',[r for r in sonic if r['task']==task],'actual','SONIC mode0 / native deployment'))
 curves.append(('actual',rr,'actual','Joint shared tracker'))
 for key,rows,field,label in curves:
  means=[];low=[];high=[];counts=[]
  for c in commands:
   values=[r[field]['quantity'] for r in rows if abs(r['command']-c)<1e-7 and r[field] is not None];counts.append(len(values));means.append(np.mean(values) if values else np.nan);low.append(np.quantile(values,.1) if values else np.nan);high.append(np.quantile(values,.9) if values else np.nan)
  line,=ax.plot(commands,means,color=colors[key],lw=1.8 if key=='actual' else 1.4,ls='--' if key in ['human','baseline','sonic'] else '-',marker='o',ms=3,label=label)
  if task==tasks[0]:handles.append(line)
  if key in ['actual','baseline','sonic']:ax.fill_between(commands,low,high,color=colors[key],alpha=.10,linewidth=0)
  if key=='actual':
   denom=len(rr)//len(commands)
   for c,q,n in zip(commands,means,counts):
    if np.isfinite(q):ax.annotate(f'{n}/{denom}',(c,q),xytext=(0,7),textcoords='offset points',ha='center',fontsize=7,color=colors[key])
 s=summary[task]['actual'];ax.set_title(f'{title} | observed {s["measurable"]}/{len(rr)}',loc='left',fontsize=11,fontweight='bold');ax.set_xlabel(f'Command ({unit})');ax.set_ylabel(f'Human-equivalent quantity ({unit})');ax.set_xticks(commands);ax.set_xticklabels([f'{c:.3g}' for c in commands]);ax.margins(x=.06,y=.16)
ns=len({r['source'] for r in x if r['task']==tasks[0]})
if a.tasks:
 for extra in list(axes.flat)[len(tasks):]:extra.axis('off')
 fig.legend(handles=handles,loc='lower center',bbox_to_anchor=(.5,.04),frameon=False,ncol=3,fontsize=9)
 fig.text(.5,.018,f'{a.scope} | {ns} sources ×5 commands per task | Bands: P10–P90, not CIs | Failures stay blank',ha='center',fontsize=8.5)
else:
 legend=axes.flat[-1];legend.axis('off');legend.legend(handles=handles,loc='lower left',frameon=False,bbox_to_anchor=(0,.01),fontsize=9.)
 legend.text(0,.95,a.scope.upper(),transform=legend.transAxes,weight='bold',fontsize=13,color='#233E53');legend.text(0,.81,f'11 tasks | {ns} sources × 5 commands\nAll {len(x)} requests retained.\nBands: source P10–P90 (not CIs).\nLabels: observed / planned.\nCompletion does not imply task success.\nMissing: blank. Angles: radians.\nSONIC: original model and PD contract.',va='top',transform=legend.transAxes,fontsize=9,linespacing=1.15)
fig.suptitle('Command response through generation, retargeting and physics',fontsize=15 if a.tasks else 17,weight='bold',y=.992);fig.tight_layout(rect=(0,.13 if a.tasks else 0,1,.962 if a.tasks else .982),h_pad=2,w_pad=2);fig.savefig(a.output/'command_response.png',dpi=170);fig.savefig(a.output/'command_response.pdf');fig.savefig(a.output/'command_response.svg');plt.close(fig)
(a.output/'figure_protocol.json').write_text(json.dumps(dict(candidate=str(a.candidate),baseline=str(a.baseline),sonic=str(a.sonic) if a.sonic else None,scope=a.scope,tasks=tasks,source_count_per_task=ns,planned=sum(r['task'] in tasks for r in x),aggregator='mean and source P10/P90, no imputation',physical_failures='blank quantity and penalized in accompanying all-request tables'),indent=2));print(a.output)
