import json,csv,math,hashlib
from pathlib import Path
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/table_demo_20261005';assert (D/'final_pipeline_complete.json').exists();out=D/'report';out.mkdir(exist_ok=False);summaries={k:json.loads((D/f'final_test/{k}_summary.json').read_text()) for k in ['unified','original_tracker']};metrics={};rows=[]
for name,s in summaries.items():
 assert s['processed']==s['planned']==32
 audits=[json.loads((Path(r['scene']['source'])/name/'audit.json').read_text()) for r in s['results']]
 assert all(x['success']==r['success'] for x,r in zip(audits,s['results']))
 metrics[name]=dict(n=32,successes=s['successes'],physical_complete=s['complete'],mean_root_error_m=s['mean_root_error'],mean_hand_error_m=s['mean_hand_error'],median_contact_s=float(np.median([x['longest_top_contact_s'] for x in audits])),min_contact_s=min(x['longest_top_contact_s'] for x in audits),max_contact_force_N=max(x['force_N']['max'] for x in audits),max_penetration_m=max(x['max_top_penetration_m'] for x in audits),failed_indexes=[r['scene']['index'] for r in s['results'] if not r['success']])
 for r,au in zip(s['results'],audits):
  sc=r['scene'];z=np.load(Path(sc['source'])/name/'actual.npz');goal=np.asarray(sc['goal_xy']);end=z['root'][-1,:2];generated=np.asarray(sc['goal_generation']['xy'][0])*sc['distance_robot_m']/sc['distance_human_m'];rows.append(dict(method=name,index=sc['index'],seed=sc['seed'],distance_m=sc['distance_robot_m'],direction_deg=np.degrees(sc['direction_rad']),generated_distance_m=np.linalg.norm(generated),generated_direction_deg=np.degrees(np.arctan2(generated[1],generated[0])),actual_distance_m=np.linalg.norm(end),actual_direction_deg=np.degrees(np.arctan2(end[1],end[0])),root_error_m=r['goal_error_m'],hand_error_m=r['palm_target_error_m'],longest_contact_s=au['longest_top_contact_s'],success=r['success']))
with (out/'final_32_paired.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
fig,axs=plt.subplots(1,3,figsize=(14,4.5));colors={'original_tracker':'#d48747','unified':'#098c83'};labels={'original_tracker':'Original shared tracker','unified':'Interaction-tuned shared tracker'}
for name in ['original_tracker','unified']:
 rr=[r for r in rows if r['method']==name]
 for ax,x,y in [(axs[0],'distance_m','actual_distance_m'),(axs[1],'direction_deg','actual_direction_deg')]:ax.scatter([r[x] for r in rr],[r[y] for r in rr],s=23,alpha=.8,color=colors[name],label=labels[name])
axs[0].plot([.8,1.8],[.8,1.8],'k--',lw=1);axs[0].set(xlabel='Requested stopping distance (m)',ylabel='Actual final root radius (m)',title='Distance preserved in physics')
axs[1].plot([-30,30],[-30,30],'k--',lw=1);axs[1].set(xlabel='Requested direction (degrees)',ylabel='Actual final root direction (degrees)',title='Direction preserved in physics')
orig=summaries['original_tracker']['results'];new=summaries['unified']['results'];assert [r['scene']['seed'] for r in orig]==[r['scene']['seed'] for r in new]
for old,r in zip(orig,new):axs[2].plot([0,1],[100*old['palm_target_error_m'],100*r['palm_target_error_m']],color='#8b9aa7',alpha=.3,lw=.7)
for j,name in enumerate(['original_tracker','unified']):axs[2].scatter(np.full(32,j),[100*r['palm_target_error_m'] for r in summaries[name]['results']],s=20,color=colors[name]);axs[2].scatter(j,100*metrics[name]['mean_hand_error_m'],marker='_',s=400,color='black',zorder=5)
axs[2].set(xticks=[0,1],xticklabels=['Original','Interaction-tuned'],ylabel='Final hand-target error (cm)',title='Same 32 references / paired scenes');axs[2].set_xlim(-.3,1.3)
for ax in axs:ax.grid(alpha=.15)
axs[0].legend(fontsize=8);fig.suptitle('Frozen-policy physical evaluation on 32 fresh random table layouts',fontsize=14);fig.text(.5,.018,'Known table pose · 0.85–1.75 m stopping radius · ±28.6° · 0.80 m tabletop · identical generator / GMR / scene-aware IK for both trackers',ha='center',fontsize=8);fig.tight_layout(rect=[0,.06,1,.94]);fig.savefig(out/'physical_results.png',dpi=180);fig.savefig(out/'physical_results.pdf')
goal=json.loads((D/'goal_command_audit/audit.json').read_text());cpu=json.loads((D/'cpu_final/audit.json').read_text());parity=json.loads((D/'frozen/parity_audit.json').read_text());final=dict(physical=metrics,generator=goal['summary'],cpu=dict(success=cpu['success'],root_error_m=cpu['root_error_m'],hand_error_m=cpu['palm_error_m'],contact_s=cpu['longest_top_contact_s']),parity_max_action_error=parity['max_action_error'],sweep=json.loads((D/'command_sweep/complete.json').read_text()),public_random_smoke=json.loads((D/'entrypoint_random_58005000/unified_summary.json').read_text())['successes'],scope='Conditional on known simulator pose, fixed table height and prompt templates, limited front sector, nominal physics; generated reference uses command-space numerical refinement and explicit reach IK. No vision or hardware claim.')
(out/'summary.json').write_text(json.dumps(final,indent=2));print(json.dumps(final,indent=2))
