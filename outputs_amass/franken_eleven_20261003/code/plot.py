import argparse,json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from audit_results import OUT,TASKS,RANGES,TOLS,native_metrics
LABELS=['Raise hand','Forward reach','Strike speed','Wave amplitude','Right turn','Right sidestep','Backward walk','Right kick','Jump height','Held forward lean','Forward walk (path speed)']
UNITS=['m','m','m/s','m','rad','m','m/s','m','m','rad','m/s']
COLORS={'old_human':'#b48457','human':'#748494','g1':'#188db6','mode0':'#e17737','mode2':'#36976a'}
def line(ax,rows,key,color,label,band=True,ls='-'):
    xs=sorted(set(r['command'] for r in rows));ys=[];low=[];high=[];ns=[]
    for x in xs:
        vals=[r[key]['quantity'] for r in rows if r['command']==x and r[key] is not None];ns.append(len(vals));ys.append(np.mean(vals) if vals else np.nan)
        low.append(np.quantile(vals,.1) if vals else np.nan);high.append(np.quantile(vals,.9) if vals else np.nan)
    ax.plot(xs,ys,marker='o',ms=3,lw=1.7,color=color,label=label,ls=ls)
    if band:ax.fill_between(xs,low,high,color=color,alpha=.10,lw=0)
    if key=='mode0':
        for x,y,n in zip(xs,ys,ns):
            if np.isfinite(y):ax.annotate(f'{n}/16',(x,y),xytext=(2,6),textcoords='offset points',fontsize=6,color=color)
def figure(rows,generation=False):
    fig,axes=plt.subplots(4,3,figsize=(16,14));axes=axes.ravel()
    for i,task in enumerate(TASKS):
        ax=axes[i];rr=[r for r in rows if r['task']==task];lo,hi=RANGES[i]
        ax.plot([lo,hi],[lo,hi],'--',color='#95a2af',lw=1,label='Target y=x')
        if generation:
            line(ax,rr,'old_human',COLORS['old_human'],'Existing weights',ls='--');line(ax,rr,'human',COLORS['g1'],'Finetuned task adapter')
            title=LABELS[i]
        else:
            line(ax,rr,'human',COLORS['human'],'Human generated reference',ls='--');line(ax,rr,'g1',COLORS['g1'],'G1 converted reference')
            line(ax,rr,'mode0',COLORS['mode0'],'SONIC mode0 actual');line(ax,rr,'mode2',COLORS['mode2'],'SONIC mode2 actual (direct SMPL)',ls='--')
            a=sum(r['mode0'] is not None for r in rr);b=sum(r['mode2'] is not None for r in rr);title=f'{LABELS[i]} | M0 {a}/80, M2 {b}/80'
        ax.set_title(title,fontsize=10,fontweight='bold',loc='left');ax.set_xlabel(f'Command ({UNITS[i]})',fontsize=9);ax.set_ylabel(f'Human-equivalent Q ({UNITS[i]})',fontsize=9);ax.tick_params(labelsize=8);ax.grid(alpha=.14)
        ax.set_xticks(np.linspace(lo,hi,5))
    ax=axes[-1];ax.axis('off')
    ax.text(0,.98,'FROZEN CONFIRMATION',fontsize=15,fontweight='bold',va='top',transform=ax.transAxes)
    note='11 tasks | 16 prompt/noise pairs x 5 commands\n4 prompt templates x 4 held-out noise seeds\nPrompts shared with finetuning; no unseen-text claim\nHuman-equivalent metres; angles remain radians\nP10-P90 source bands, NOT confidence intervals\nNo test-time editing or simulator parameter search'
    if not generation:note+='\nLabels: measurable/16; missing excluded, E_all=1\nDefault release (not v1.1); green=mode2, not PULSE'
    ax.text(0,.83,note,fontsize=8,linespacing=1.1,va='top',transform=ax.transAxes)
    handles,labels=axes[0].get_legend_handles_labels();ax.legend(handles,labels,loc='lower left',fontsize=8,frameon=False)
    fig.suptitle('FrankenMotion numeric command response: '+('before / after adapter finetuning (reference only)' if generation else 'human reference / G1 conversion / physical execution'),fontsize=17,y=.995)
    fig.tight_layout(rect=(0,0,1,.98),h_pad=1.9,w_pad=1.5)
    return fig
def main():
    ap=argparse.ArgumentParser();ap.add_argument('--generation-only',action='store_true');a=ap.parse_args();out=OUT/'delivery';out.mkdir(exist_ok=True)
    if a.generation_only:
        new=json.loads((OUT/'generated/task_adapter_manifest.json').read_text());old=json.loads((OUT/'generated/existing_adapter_manifest.json').read_text());old={(r['source'],r['command_index']):r for r in old}
        rows=[dict(task=r['task'],source=r['source'],command=r['command'],human=native_metrics(r),old_human=native_metrics(old[(r['source'],r['command_index'])])) for r in new]
        fig=figure(rows,True);name='generation_before_after'
    else:rows=json.loads((out/'audited_results.json').read_text());fig=figure(rows);name='franken_eleven_response'
    fig.savefig(out/(name+'.png'),dpi=180);fig.savefig(out/(name+'.svg'));plt.close(fig)
    print('Saved',name)
if __name__=='__main__':main()
