"""Publication-style response curves from frozen, audited confirmation only."""
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path('/home/pku/frankenmotion/outputs_amass/franken_improve_20261003');OUT=ROOT/'final_delivery'
rows=json.loads((OUT/'audited_results.json').read_text());audit=json.loads((OUT/'trajectory_audit.json').read_text());assert audit['records_checked']==3520
TASKS=['raise_hand','reach','strike','wave','turn','sidestep','back_walk','kick','jump','lean','walk'];UNITS=['m','m','m/s','m','rad','m','m/s','m','m','rad','m/s'];NAMES=['Raise hand','Forward reach','Strike speed','Wave amplitude','Right turn','Right sidestep','Backward walk','Right kick','Jump height','Held forward lean','Forward walk (path speed)']
for paired in [False,True]:
 fig,axes=plt.subplots(4,3,figsize=(16,14));axlist=axes.ravel()
 series=[('baseline_direction','#96a1ac','Prior adapter + direction IK'),('baseline_gmr','#58a1b9','Prior adapter + GMR'),('sonic','#df773d','New adapter + GMR + SONIC'),('selected','#31836b','Frozen selected pipeline')] if paired else [('human','#87929d','Generated human reference'),('g1','#168caf','GMR G1 reference'),('sonic','#df773d','SONIC v1.1 actual'),('beyondmimic','#31836b','BeyondMimic bank actual')]
 for i,task in enumerate(TASKS):
  ax=axlist[i];rr=[r for r in rows if r['task']==task];xs=sorted({r['command'] for r in rr});ax.plot(xs,xs,'--',lw=1,color='#acb6bf',label='Target y=x')
  for key,color,label in series:
   values=[[r[key]['quantity'] for r in rr if r['command']==x and r[key] is not None] for x in xs];mean=[np.mean(x) if x else np.nan for x in values];ax.plot(xs,mean,'o-' if key!='human' else 'o--',lw=1.5,ms=3,color=color,label=label)
   if key not in ['human','g1']:ax.fill_between(xs,[np.quantile(x,.1) if x else np.nan for x in values],[np.quantile(x,.9) if x else np.nan for x in values],color=color,alpha=.09,lw=0)
  n1=sum(r['sonic'] is not None for r in rr);n2=sum(r['beyondmimic'] is not None for r in rr);chosen=rr[0]['selected_method'];title=NAMES[i] if paired else f'{NAMES[i]} | S {n1}/80, B {n2}/80'
  ax.set_title(title,loc='left',fontsize=10,fontweight='bold');ax.set_xlabel('Command ('+UNITS[i]+')',fontsize=9);ax.set_ylabel('Human-equivalent Q ('+UNITS[i]+')',fontsize=9);ax.set_xticks(xs);ax.tick_params(labelsize=8);ax.grid(alpha=.14)
 panel=axlist[-1];panel.axis('off');panel.text(0,.98,'FROZEN CONFIRMATION\n\n11 tasks x 16 sources x 5 commands\n4 shared prompts x 4 held-out noise seeds\nP10-P90 source bands, not confidence intervals\nMissing excluded from curves; E_all assigns 1\nSame standing entry and physical fall thresholds\nNative actuator/simulator contracts differ\nB = preselected task-specific checkpoint bank\nNo hardware execution; no unseen-text claim',va='top',fontsize=8,linespacing=1.15);handles,labels=axlist[0].get_legend_handles_labels();panel.legend(handles,labels,loc='lower left',fontsize=8,frameon=False)
 fig.suptitle('FrankenMotion command response: '+('paired before / after' if paired else 'reference / retarget / physical execution'),fontsize=17,y=.995);fig.tight_layout(rect=[0,0,1,.98],h_pad=1.8,w_pad=1.5);name='paired_improvement' if paired else 'command_response_confirmation';fig.savefig(OUT/(name+'.png'),dpi=180);fig.savefig(OUT/(name+'.svg'));plt.close(fig)
print('Saved both 11-task confirmation figures')
