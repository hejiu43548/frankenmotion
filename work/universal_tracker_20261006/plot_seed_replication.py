from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'outputs/universal_tracker_20261006';d=json.loads((D/'report_data.json').read_text());assert d['replication']['status']['stage']=='complete';out=D/'figures/seed_replication';out.mkdir(exist_ok=True)
tasks=['raise_hand','reach','strike','wave','turn','sidestep','back_walk','kick','jump','lean','walk'];labels=['Raise','Reach','Strike','Wave','Turn','Side step','Back walk','Kick','Jump','Lean','Walk'];sets=[('Previous shared tracker',d['controllers']['broad']['per_task'],'#876CA9'),('Expanded seed 6106 (frozen main)',d['controllers']['candidate']['per_task'],'#D56A35'),('Expanded seed 6107 (fixed last)',d['replication']['results']['per_task'],'#368E9D')]
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'pdf.fonttype':42,'svg.fonttype':'none','axes.spines.top':False,'axes.spines.right':False})
fig,ax=plt.subplots(figsize=(11.5,4.1));x=np.arange(len(tasks));width=.25
for i,(label,values,color) in enumerate(sets):
 y=[values[t]['actual']['joint_pass']/80*100 for t in tasks];ax.bar(x+(i-1)*width,y,width,label=label,color=color)
ax.set_xticks(x,labels,rotation=20);ax.set_ylim(0,105);ax.set_ylabel('Event + numeric success (%)');ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True);ax.set_title('Same 880 frozen references | Two expansion-stage seeds, shared warm start',loc='left',weight='bold');fig.legend(loc='lower center',bbox_to_anchor=(.5,.015),ncol=3,frameon=False,fontsize=9);fig.tight_layout(rect=(0,.10,1,1));
for suffix in ['png','pdf','svg']:fig.savefig(out/('seed_replication.'+suffix),dpi=180)
(out/'protocol.json').write_text(json.dumps(dict(scope='Fixed main checkpoint versus predeclared independent continuation seed. Shared v4 initialization; no seed/checkpoint reselection. Not full-pipeline variance or a training-seed confidence interval.',counts={label:{t:values[t]['actual']['joint_pass'] for t in tasks} for label,values,color in sets}),indent=2));print(out)
