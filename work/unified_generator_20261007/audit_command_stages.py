import json,csv
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path('/home/pku/frankenmotion/outputs_amass/unified_generator_20261007');out=D/'command_stage_audit';out.mkdir(exist_ok=True)
rows=json.loads((D/'general_evaluation/unified_final/audited_results.json').read_text());s=json.loads((D/'general_evaluation/unified_final/eleven_summary.json').read_text());old=json.loads((D/'general_evaluation/teacher_final/eleven_summary.json').read_text());tasks=list(s);tols=[.02,.02,.1,.015,.08,.06,.05,.05,.04,.06,.05];units=['m','m','m/s','m','rad','m','m/s','m','m','rad','m/s'];fig,axs=plt.subplots(4,3,figsize=(15,15));records=[]
for i,t in enumerate(tasks):
 rr=[r for r in rows if r['task']==t];assert len(rr)==80;ax=axs.flat[i];cmds=sorted({r['command'] for r in rr});record={'task':t,'tolerance':tols[i],'unit':units[i]}
 for stage,color,label in [('human','#3978bb','Human reference'),('g1','#e68a29','G1 retargeted reference'),('actual','#3b986c','G1 physics (same tracker)')]:
  means=[];low=[];high=[]
  for c in cmds:
   v=[r[stage]['quantity'] for r in rr if r['command']==c and r[stage] is not None];means.append(float(np.mean(v)));low.append(np.quantile(v,.1));high.append(np.quantile(v,.9))
  ax.plot(cmds,means,'o-',color=color,label=label,markersize=4);ax.fill_between(cmds,low,high,color=color,alpha=.12)
  groups=[[r for r in rr if r['source']==src] for src in sorted({r['source'] for r in rr})];monotonic=[]
  for group in groups:
   group=sorted(group,key=lambda r:r['command'])
   if all(r[stage] is not None for r in group):monotonic.append(bool(np.all(np.diff([r[stage]['quantity'] for r in group])>0)))
  record[stage]=dict(s[t][stage],strict_monotonic_sources=sum(monotonic),complete_sources=len(monotonic),mean_per_command=means,commands=cmds)
 passing=lambda r,k:r[k] is not None and r[k]['event_pass'] and abs(r[k]['quantity']-r['command'])<=tols[i]
 record['human_pass_lost_in_retarget']=sum(passing(r,'human') and not passing(r,'g1') for r in rr);record['human_fail_gained_in_retarget']=sum(not passing(r,'human') and passing(r,'g1') for r in rr)
 record['retarget_quantity_change_mae']=float(np.mean([abs(r['human']['quantity']-r['g1']['quantity']) for r in rr]));record['retarget_quantity_change_bias']=float(np.mean([r['g1']['quantity']-r['human']['quantity'] for r in rr]));records.append(record)
 ax.plot(cmds,cmds,'--',color='#888888',label='Ideal');ax.set_title(f'{t} | H {s[t]["human"]["joint_pass"]}/80 -> G1 {s[t]["g1"]["joint_pass"]}/80');ax.set_xlabel('Command ('+units[i]+')');ax.set_ylabel('Measured quantity ('+units[i]+')');ax.grid(alpha=.2)
axs.flat[11].axis('off');h,l=axs.flat[0].get_legend_handles_labels();axs.flat[11].legend(h,l,loc='upper left');axs.flat[11].text(0,.65,'11 tasks x 4 prompts x 4 fresh seeds x 5 commands\nFixed unified generator; no final-set fitting\nHuman-equivalent length/speed; angles in radians\nBands: P10-P90, not confidence intervals\nPass = numeric tolerance AND motion event\nPhysics Q omits 3 incomplete runs; pass counts retain them\nG1 retarget includes frozen raise_hand/lean refinement\nFixed prompts: does not test unseen-text generalization',va='top',fontsize=10,linespacing=1.8)
fig.tight_layout();fig.savefig(out/'command_stages.png',dpi=180);fig.savefig(out/'command_stages.pdf');plt.close(fig)
agg={k:{f:sum(s[t][k][f] for t in tasks) for f in ['planned','measurable','numeric_pass','event_pass','joint_pass']} for k in ['human','g1','actual']};agg['previous']={k:sum(old[t][k]['joint_pass'] for t in tasks) for k in ['human','g1','actual']}
(out/'summary.json').write_text(json.dumps(dict(aggregate=agg,tasks=records,metadata_note='Some inherited raw result rows incorrectly say split=development. Generation protocol and seeds identify the frozen final split; this audit uses all880 final samples.'),indent=2))
flat=[]
for r in rows:
 row={k:r[k] for k in ['task','source','seed','command','command_index']}
 for k in ['human','g1','actual']:
  row[k+'_Q']=r[k]['quantity'] if r[k] else None;row[k+'_event']=r[k]['event_pass'] if r[k] else False
 flat.append(row)
with (out/'all_880.csv').open('w') as f:
 w=csv.DictWriter(f,fieldnames=flat[0]);w.writeheader();w.writerows(flat)
print(json.dumps(agg));print([(v['task'],v['human']['strict_monotonic_sources'],v['g1']['strict_monotonic_sources'],round(v['retarget_quantity_change_bias'],4)) for v in records])
