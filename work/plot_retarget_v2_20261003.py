import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path('/home/pku/frankenmotion/outputs_amass/franken_improve_20261003/final_delivery_v2');rows=json.loads((OUT/'audited_results.json').read_text());audit=json.loads((OUT/'audit.json').read_text());assert audit['physical_rollouts']==320
fig,axes=plt.subplots(1,2,figsize=(12,5.8))
for ax,task,title,unit in zip(axes,['raise_hand','lean'],['Raise hand','Held forward lean'],['m','rad']):
 rr=[r for r in rows if r['task']==task];xs=sorted({r['command'] for r in rr});ax.plot(xs,xs,':',color='.5',label='Target')
 for key,label,color,style in [('human','Generated human','#87929d','--'),('g1_w0','Original GMR reference','#b690cf','--'),('actual_w0','Original GMR + SONIC','#df773d','-'),('g1_w4','Task-constrained GMR reference','#56b9c0','--'),('actual_w4','Task-constrained GMR + SONIC','#248451','-')]:
  values=[[r[key]['quantity'] for r in rr if r['command']==x and r[key] is not None] for x in xs];ax.plot(xs,[np.mean(x) if x else np.nan for x in values],style,marker='o',ms=4,color=color,label=label)
  if key.startswith('actual'):ax.fill_between(xs,[np.quantile(x,.1) if x else np.nan for x in values],[np.quantile(x,.9) if x else np.nan for x in values],color=color,alpha=.1)
 ax.set_title(title,fontweight='bold');ax.set_xlabel(f'Command ({unit})');ax.set_ylabel(f'Human-equivalent Q ({unit})');ax.grid(alpha=.2)
handles,labels=axes[0].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',bbox_to_anchor=(.5,.08),ncol=3,frameon=False,fontsize=9);fig.suptitle('Separate v2 confirmation: source-derived retarget constraints',fontsize=15);fig.text(.5,.02,'160 fresh-noise requests, 320 paired physical rollouts | Same frozen generator and SONIC | P10–P90 source bands, not CI\nFour shared prompt templates; no unseen-text claim. No target-Q compensation or generation-output editing.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.22,1,.94]);fig.savefig(OUT/'retarget_v2_response.png',dpi=180);fig.savefig(OUT/'retarget_v2_response.svg')
