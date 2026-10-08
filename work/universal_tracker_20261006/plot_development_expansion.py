"""Expanded-corpus continuation on development data; no uncertainty from one run."""
import argparse,json
from pathlib import Path
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--audit',required=True);p.add_argument('--output',required=True);p.add_argument('--selected');a=p.parse_args();data=json.loads(Path(a.audit).read_text());rows=[]
for r in data['results']:
 if r['name']=='v4_3000':rows.append((0,r))
 elif r['name'].startswith('v5_') and r.get('expansion_selection_complete'):
  suffix=r['name'][3:];step=5999 if suffix=='final' else int(suffix)
  rows.append((step,r))
rows.sort(key=lambda x:x[0]);assert len(rows)>1
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':10,'axes.spines.top':False,'axes.spines.right':False,'pdf.fonttype':42,'svg.fonttype':'none'})
fig,axs=plt.subplots(1,3,figsize=(12,3.5));x=np.array([s for s,r in rows]);values=[[r['eleven_joint_passes']/110*100 for s,r in rows],[r['natural_accurate_complete']/r['natural_requests']*100 for s,r in rows],[r['walk_torso_jitter_ratio'] for s,r in rows]]
for ax,y,title,label in zip(axs,values,['11 command tasks (GPU, N=110)','Natural-source validation (CPU, N=57)','Historical table scenes (CPU, N=6)'],['Event + numeric pass (%)','Accurate completion (%)','Torso high-frequency RMS / stable']):
 ax.plot(x,y,'o-',color='#d46b3d',lw=2,ms=5);ax.axhline(y[0],ls=':',lw=1.3,color='#55768e',label='Warm-start checkpoint');ax.set_title(title,fontsize=11);ax.set_xlabel('Expanded-corpus PPO iteration');ax.set_ylabel(label);ax.grid(alpha=.17)
 for step,r in rows:
  if r['name']==a.selected:ax.axvline(step,color='#4b927a',lw=1.1,ls='--',label='Frozen selection')
axs[0].set_ylim(0,100);axs[1].set_ylim(0,100)
axs[2].axhline(1.5,color='#a84747',ls='--',lw=1,label='Smoothness gate')
for step,r in rows:axs[2].annotate(str(r['table_success'])+'/6',(step,r['walk_torso_jitter_ratio']),xytext=(0,9),textcoords='offset points',ha='center',fontsize=8)
axs[2].set_ylim(top=max(1.7,max(values[2])+.2));handles,labels=axs[2].get_legend_handles_labels();fig.legend(handles,labels,loc='lower center',ncol=3,frameon=False,bbox_to_anchor=(.5,.035),fontsize=9);fig.suptitle('Expanded joint training: one policy, one training seed',y=1.025,fontsize=13);fig.text(.5,-.015,'0 = v4 warm start; table labels show full-flow success. Development curves are not frozen-test results.',ha='center',fontsize=9,color='#596773');fig.tight_layout(rect=[0,.13,1,1]);out=Path(a.output);out.mkdir(parents=True,exist_ok=True)
for ext in ['png','pdf','svg']:fig.savefig(out/('expanded_training.'+ext),dpi=200,bbox_inches='tight')
(out/'plot_data.json').write_text(json.dumps(dict(scope=__doc__,selected=a.selected,points=[dict(iteration=s,record=r) for s,r in rows]),indent=2))
