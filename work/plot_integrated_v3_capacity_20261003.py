from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
OUT=Path('/home/pku/frankenmotion/outputs_amass/franken_improve_20261003/final_delivery_integrated_v3_capacity256');rows=json.loads((OUT/'audited_results.json').read_text());assert json.loads((OUT/'audit.json').read_text())['checks']==1120
TASKS=['raise_hand','reach','strike','wave','turn','sidestep','back_walk','kick','jump','lean','walk'];NAMES=['Raise hand','Forward reach','Strike speed','Wave amplitude','Right turn','Right sidestep','Backward walk','Right kick','Jump height','Held forward lean','Forward walk (path speed)'];UNITS=['m','m','m/s','m','rad','m','m/s','m','m','rad','m/s']
fig,axs=plt.subplots(4,3,figsize=(16,14))
for ax,t,name,unit in zip(axs.flat,TASKS,NAMES,UNITS):
 a=[r for r in rows if r['task']==t];xs=sorted({r['command'] for r in a});ax.plot(xs,xs,':',color='.65',lw=1,label='Target y=x')
 for key,label,color,style in [('human','Integrated human reference','#87929d','--'),('g1','Integrated G1 reference','#168caf','--'),('v1_actual','V1 selected pipeline, same new seeds','#df773d','-'),('actual','Integrated V3 actual','#31836b','-')]:
  vs=[[r[key]['quantity'] for r in a if r['command']==x and r[key] is not None] for x in xs];ax.plot(xs,[np.mean(v) if v else np.nan for v in vs],style,marker='o',ms=3,color=color,lw=1.5,label=label)
  if key in ['actual','v1_actual']:ax.fill_between(xs,[np.quantile(v,.1) if v else np.nan for v in vs],[np.quantile(v,.9) if v else np.nan for v in vs],color=color,alpha=.09)
 n=sum(r['actual'] is not None for r in a);ax.set_title(f'{name} | {n}/80 complete',loc='left',fontweight='bold',fontsize=10);ax.set_xlabel(f'Command ({unit})',fontsize=9);ax.set_ylabel(f'Human-equivalent Q ({unit})',fontsize=9);ax.set_xticks(xs);ax.tick_params(labelsize=8);ax.grid(alpha=.15)
ax=axs.flat[-1];ax.axis('off');ax.text(0,.98,'V3 / CONTACT CAPACITY 256\n\n11 tasks x 16 sources x 5 commands\n880 fresh-noise requests; 1120 unique rollouts\nV1/V3 share execution for 8 unchanged tasks\n3 changed tasks get separate paired executions\nRaise/lean: source-derived retarget constraints\nKick: original generator adapter restored\nSame frozen per-task controller routing\nP10-P90 source bands, not confidence intervals\nMissing excluded from curves; E_all assigns 1\nShared prompt templates; no hardware execution',va='top',fontsize=8,linespacing=1.15);h,l=axs.flat[0].get_legend_handles_labels();ax.legend(h,l,loc='lower left',frameon=False,fontsize=8);fig.suptitle('FrankenMotion to G1: integrated command response',fontsize=17,y=.995);fig.tight_layout(rect=[0,0,1,.98],h_pad=1.8,w_pad=1.5);fig.savefig(OUT/'integrated_v3_response.png',dpi=180);fig.savefig(OUT/'integrated_v3_response.svg');print('Saved integrated V3 11-task plot')
