import json,csv
from pathlib import Path
import numpy as np,matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
D=Path('/home/pku/frankenmotion/outputs_amass/table_demo_20261005/goal_command_audit');rows=json.loads((D/'results.json').read_text());paths=np.load(D/'paths.npz')['xy'];fig,axes=plt.subplots(1,3,figsize=(15,5.0));colors={'existing_speed_yaw':'#8392a5','learned_goal':'#078c83'};names={'existing_speed_yaw':'Existing speed / yaw controls','learned_goal':'Learned distance / direction'}
for ax,xkey,ykey,unit in [(axes[0],'distance_robot_m','actual_distance_m',1),(axes[1],'direction_rad','actual_direction_rad',180/np.pi)]:
 xs=sorted({r[xkey] for r in rows})
 for kind in colors:
  data=[[r[ykey]*unit for r in rows if r['kind']==kind and r[xkey]==x] for x in xs];lo,med,hi=np.quantile(data,[.1,.5,.9],axis=1);ax.plot(np.array(xs)*unit,med,'o-',color=colors[kind],label=names[kind]);ax.fill_between(np.array(xs)*unit,lo,hi,color=colors[kind],alpha=.13)
 ax.plot(np.array(xs)*unit,np.array(xs)*unit,'k--',lw=1,label='Requested');ax.grid(alpha=.18)
axes[0].set(xlabel='Requested distance (robot-equivalent m)',ylabel='Generated travel distance (m)',title='Distance response');axes[1].set(xlabel='Requested direction (degrees)',ylabel='Generated travel direction (degrees)',title='Direction response');axes[0].legend(fontsize=8)
for i,r in enumerate(rows):
 if r['prompt']==0 and r['seed']==56005000 and r['distance_robot_m']>1.3 and r['distance_robot_m']<1.5 and r['direction_rad'] in [-.5,0,.5]:
  path=paths[i];axes[2].plot(path[:,0],path[:,1],color=colors[r['kind']],alpha=.9,lw=1.7)
  if r['kind']=='learned_goal':
   target=r['distance_robot_m']*np.array([np.cos(r['direction_rad']),np.sin(r['direction_rad'])]);axes[2].scatter(*target,marker='x',s=70,c='#142231')
axes[2].set(xlabel='Forward (m)',ylabel='Lateral (m)',title='Same noise, different goal directions');axes[2].axis('equal');axes[2].grid(alpha=.18)
fig.suptitle('Goal conditioning: matched prompts, noise and legacy controls',fontsize=15);fig.text(.5,.047,'120 paired commands · four cached prompts × two fresh seeds × three distances × five directions · bands: P10–P90 · generation only',ha='center',fontsize=9);fig.text(.5,.015,'Ablation with identical demo-derived speed/yaw profiles; some low-speed commands extrapolate beyond the legacy task-adapter validation range.',ha='center',fontsize=8)
fig.tight_layout(rect=[0,.09,1,.94]);fig.savefig(D/'goal_command_responses.png',dpi=180);fig.savefig(D/'goal_command_responses.pdf')
with (D/'goal_command_responses.csv').open('w') as f:
 writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
