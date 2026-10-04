"""Plot audited command responses; never silently discard failed requests."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/home/pku/frankenmotion');U=R/'outputs_amass/franken_unified_20261004'
sys.path.insert(0,str(R/'outputs_amass/franken_eleven_20261003/code'))
from audit_results import TASKS
p=argparse.ArgumentParser();p.add_argument('--names',nargs='+',required=True);p.add_argument('--baseline',default='routed_baseline_validation');p.add_argument('--output',default='development_responses');p.add_argument('--labels',nargs='+');a=p.parse_args();assert a.labels is None or len(a.labels)==len(a.names);display=dict(zip(a.names,a.labels or a.names))
read=lambda p:json.loads(p.read_text())
base=read(U/'evaluation'/a.baseline/'audited_results.json');models={};split=read(U/'evaluation'/a.baseline/'protocol.json').get('split','development_validation')
for name in a.names:
    folder=U/'evaluation'/name; audit=read(folder/'audit.json')
    assert audit['unique_checkpoints']==1 and audit['raw_max_error']<1e-8
    rows=read(folder/'audited_results.json')
    identity=lambda r:(r['path'],r['seed'],r['command'])
    assert sorted(map(identity,rows))==sorted(map(identity,base))
    models[name]=rows
fig,axes=plt.subplots(4,3,figsize=(18,17));palette=['#e87722','#ae439c','#008f91','#7254ab']
for ax,task in zip(axes.flat,TASKS):
    br=[r for r in base if r['task']==task];commands=sorted({r['command'] for r in br})
    curves=[('Human reference',br,'human','#969da4','--'),('G1 reference',br,'g1','#227db5','--'),('Routed baseline',br,'actual','#399155',':')]
    curves += [(name,[r for r in rows if r['task']==task],'actual',palette[i%len(palette)],'-') for i,(name,rows) in enumerate(models.items())]
    counts=[]
    for label,rows,key,color,style in curves:
        med=[];lo=[];hi=[]
        for command in commands:
            values=[r[key]['quantity'] for r in rows if r['command']==command and r[key] is not None]
            med.append(np.median(values) if values else np.nan);lo.append(np.percentile(values,10) if values else np.nan);hi.append(np.percentile(values,90) if values else np.nan)
        ax.plot(commands,med,style,color=color,marker='o',ms=3,label=display.get(label,label))
        if key=='actual':ax.fill_between(commands,lo,hi,color=color,alpha=.08)
        if label==a.names[0]:
            for x,y in zip(commands,med):
                group=[r for r in rows if r['command']==x];n=sum(r[key] is not None for r in group)
                if np.isfinite(y):ax.annotate(f'{n}/{len(group)}',(x,y),xytext=(0,6),textcoords='offset points',ha='center',color=color,fontsize=7)
        if label in models:counts.append(f"{display.get(label,label)}: {sum(r[key] is not None for r in rows)}/{len(rows)} complete")
    ax.plot(commands,commands,color='#aeb2b6',lw=1,ls=':',label='Identity')
    unit='rad' if task in ['turn','lean'] else ('m/s' if task in ['strike','walk','back_walk'] else 'm');ax.set_title(task.replace('_',' ').title(),fontweight='bold');ax.set_xlabel(f'Requested command ({unit})');ax.set_ylabel(f'Human-equivalent quantity ({unit})');ax.grid(alpha=.18)
    ax.text(.02,.98,'\n'.join(counts),transform=ax.transAxes,va='top',fontsize=7)
ax=axes.flat[-1];ax.axis('off');handles,labels=axes.flat[0].get_legend_handles_labels();ax.legend(handles,labels,loc='upper left',frameon=False,fontsize=9)
ax.text(0,.55,'Audited matched references and seeds\nMedian; bands P10-P90 over completed rollouts\nFailures remain in aggregate E_all (penalty 1)\nCompletion does not imply semantic success\nPoint labels: completed/planned for first policy.\n' + ('Independent final test; selected policy frozen first.' if split=='final_test' else 'Development selection data; not a final test.') + '',va='top',fontsize=10)
fig.suptitle('One shared tracker: command response comparison',fontsize=18);fig.tight_layout(rect=[0,0,1,.97]);dest=U/'figures';dest.mkdir(exist_ok=True)
for ext in ['png','pdf']:fig.savefig(dest/(a.output+'.'+ext),dpi=170,bbox_inches='tight')
print(dest/a.output)
