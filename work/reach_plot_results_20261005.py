"""Plot every final paired case, including failures; no success-only filtering."""
import json,argparse,csv
from pathlib import Path
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
p=argparse.ArgumentParser();p.add_argument('--folder',required=True);p.add_argument('--name',default='selected');p.add_argument('--output',required=True);a=p.parse_args();folder=Path(a.folder);out=Path(a.output);out.mkdir(parents=True,exist_ok=True);rows=json.loads((folder/'manifest.json').read_text());ref={r['index']:r for r in json.loads((folder/'command_reference_audit.json').read_text())};data=[]
for row in rows:
 m=json.loads((Path(row['source'])/a.name/'command_metrics.json').read_text());data.append(dict(index=row['index'],command=row['command'],human=ref[row['index']]['human_hold_mean'],g1_reference=ref[row['index']]['g1_reference_hold_mean'],actual=m['actual_hold_wrist_forward_human_equiv_m'],contact=m['contact_during_hold_s'],longest_hold_contact=m['longest_hold_contact_s'],lowering=m['hand_lowering_m'],exit_task=m['exit_task'],exit_distance=m['signed_exit_distance_m'],success=m['success'],complete=m['physical_complete'],layout=row['walk_source'],walk_distance=row['distance_robot_m'],walk_direction_deg=np.degrees(row['direction_rad'])))
with (out/'all_cases.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=data[0].keys());w.writeheader();w.writerows(data)
plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False});fig,axs=plt.subplots(1,3,figsize=(14,4.5));colors={'human':'#7a8798','g1_reference':'#008fb2','actual':'#dc702c'};cmds=np.array([.3,.5]);ax=axs[0]
for key,label in [('human','FrankenMotion generated'),('g1_reference','GMR G1 reference'),('actual','Physical execution')]:
 values=[np.array([r[key] for r in data if r['command']==c],dtype=float) for c in cmds];ax.errorbar(cmds,[np.nanmean(v) for v in values],yerr=[np.nanstd(v) for v in values],label=label,color=colors[key],marker='o',capsize=4,linewidth=2)
ax.plot([.26,.54],[.26,.54],'--',color='#bcc3cd',label='Ideal command');ax.set(xlim=(.26,.54),ylim=(.23,.57),xticks=cmds,xlabel='Reach command (human-equivalent m)',ylabel='Mean held wrist reach (same units)',title='Command survives the full pipeline');ax.legend(fontsize=8,loc='upper left');ax.grid(alpha=.15)
ax=axs[1]
for c,offset in [(.3,-.035),(.5,.035)]:
 vals=[r['longest_hold_contact'] for r in data if r['command']==c];ax.scatter(np.arange(len(vals))+offset,vals,label=f'{c:.1f} m command',s=45)
ax.axhline(.5,ls='--',color='#bdc5d0',label='Acceptance minimum');ax.set(xlabel='Paired scene / departure condition',ylabel='Longest continuous top contact in hold (s)',title='Real collision contact');ax.legend(fontsize=8);ax.grid(alpha=.15)
ax=axs[2]
for k,task in enumerate(['back_walk','sidestep']):
 for c,offset in [(.3,-.12),(.5,.12)]:
  vals=[r['exit_distance'] for r in data if r['exit_task']==task and r['command']==c];ax.scatter(k+offset+np.linspace(-.035,.035,len(vals)),vals,s=45,color=('#1f77b4' if c==.3 else '#ff7f0e'),label=f'{c:.1f} m reach' if k==0 else None)
ax.set(xticks=[0,1],xticklabels=['Backward departure','Side departure'],ylabel='Signed physical travel (robot m)',title='Generated departure after lowering');ax.legend(fontsize=8);ax.grid(alpha=.15)
fig.suptitle(f"Single tracker / all {len(data)} fresh cases / {sum(r['success'] for r in data)} full-stage successes",fontsize=15);fig.text(.02,.015,'Left: mean ± standard deviation across paired layouts and exits. Right: every rollout, including failures. No hardware validation.',fontsize=9,color='#596579');fig.tight_layout(rect=[0,.05,1,.92]);fig.savefig(out/'command_response.png',dpi=180);fig.savefig(out/'command_response.pdf');plt.close(fig)
summary=dict(count=len(data),complete=sum(r['complete'] for r in data),successes=sum(r['success'] for r in data),by_command={str(c):{key:float(np.nanmean(np.array([r[key] for r in data if r['command']==c],dtype=float))) for key in ['human','g1_reference','actual','contact','lowering']} for c in cmds},paired_actual_deltas=[(data[i+1]['actual']-data[i]['actual']) if data[i+1]['actual'] is not None and data[i]['actual'] is not None else None for i in range(0,len(data),2)],command_definition='Forward wrist relative to pelvis during held phase, human-equivalent m. Multiply by 1.0486437524221748/1.2701193988323212 for physical G1 wrist distance.',all_cases_included=True);(out/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary,indent=2))
