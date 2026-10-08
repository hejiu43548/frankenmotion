"""Fixed frozen-test perturbation results, not multi-seed hardware robustness."""
import argparse,json
from pathlib import Path
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--data',required=True);p.add_argument('--output',required=True);a=p.parse_args();data=json.loads(Path(a.data).read_text());profiles=['nominal','friction_0p6','mass_1p1','delay_20ms','lateral_push_40N'];labels=['Nominal','Friction x0.6','Mass x1.1','20 ms delay','40 N push'];plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'});fig,axes=plt.subplots(1,2,figsize=(10.5,3.7),sharey=True);x=np.arange(5)
for ax,metric,title in zip(axes,['complete','joint'],['Physical completion','Event + numeric success']):
 for offset,tag,label,color in [(-.18,'broad','Previous shared tracker','#8263a6'),(.18,'candidate','Frozen joint tracker','#dc783e')]:
  values=[]
  for profile in profiles:
   r=data['robustness'][tag][profile];assert r is not None;z=r['aggregate'];assert z['requests']==110;values.append(z[metric]/110*100)
  bars=ax.bar(x+offset,values,width=.34,color=color,label=label)
  for bar,v in zip(bars,values):ax.text(bar.get_x()+bar.get_width()/2,v+1.3,f'{v:.0f}',ha='center',fontsize=8)
 ax.set_xticks(x,labels,rotation=18,ha='right');ax.set_ylim(0,108);ax.set_title(title);ax.grid(axis='y',alpha=.18);ax.set_axisbelow(True)
axes[0].set_ylabel('All-request rate (%)');fig.legend(*axes[0].get_legend_handles_labels(),loc='lower center',ncol=2,frameon=False,bbox_to_anchor=(.5,-.02));fig.suptitle('Frozen reference subset | one factor changed at a time | N=110',fontsize=12,y=1.02);fig.tight_layout(rect=[0,.09,1,1]);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
for ext in ['png','pdf','svg']:fig.savefig(out/('robustness.'+ext),dpi=200,bbox_inches='tight')
