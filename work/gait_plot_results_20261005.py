import json
from pathlib import Path
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/gait_demo_20261005';out=D/'report';out.mkdir(exist_ok=True);d=json.loads((D/'development_comparison.json').read_text());labels=['Previous','Retime only','New single actor'];keys=['previous','retime_only','new_single_actor'];fig,axs=plt.subplots(1,4,figsize=(14,3.9));metrics=[('knee_amplitude_ratio','Knee amplitude / reference',100,'%'),('clearance_p95_m','Foot clearance (95th percentile)',100,'cm'),('double_support_fraction','Double support fraction',100,'%'),('mean_contact_point_slip_m_s','Contact-point slip speed',100,'cm/s')]
for ax,(key,title,factor,unit) in zip(axs,metrics):
 vals=[d['summary'][k][key]*factor for k in keys];ax.bar(labels,vals,color=['#8998a6','#e7ac64','#21a0a0']);ax.set_title(title,fontsize=10);ax.set_ylabel(unit);ax.tick_params(axis='x',labelrotation=15);ax.spines[['top','right']].set_visible(False)
 for i,v in enumerate(vals):ax.text(i,v,f'{v:.1f}',ha='center',va='bottom',fontsize=10)
 ax.set_ylim(0,max(vals)*1.22)
fig.suptitle('Matched six development scenes | same generated spatial paths | GPU physics',fontsize=13);fig.text(.5,.01,'Engineering diagnostics complement visual review; these are not validated perceptual naturalness scores.',ha='center',fontsize=9);fig.tight_layout(rect=[0,.04,1,.94]);fig.savefig(out/'gait_comparison.png',dpi=180);plt.close(fig)
f=json.loads((D/'final_results.json').read_text());fig,axs=plt.subplots(1,3,figsize=(13,3.8))
for backend,color,label in [('selected','#168e9b','GPU'),('selected_cpu','#dc8558','CPU')]:
 cases=f[backend]['cases'];xx=np.arange(32);axs[0].plot(xx,[r['result']['goal_error_m']*100 for r in cases],'.-',color=color,label=label);axs[1].plot(xx,[r['result']['palm_target_error_m']*100 for r in cases],'.-',color=color,label=label);axs[2].plot(xx,[r['result']['longest_hand_contact_s'] for r in cases],'.-',color=color,label=label)
for ax,title,unit in zip(axs,['Walking goal error','Hand target error','Longest measured top contact'],['cm','cm','s']):ax.set_title(title);ax.set_xlabel('Random scene index');ax.set_ylabel(unit);ax.spines[['top','right']].set_visible(False);ax.legend()
fig.suptitle('32 fresh random layouts | one frozen actor | actual physical rollouts');fig.tight_layout(rect=[0,0,1,.94]);fig.savefig(out/'final_results.png',dpi=180)
