"""One fixed repeated-route interface comparison; no inference across independent trials."""
from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
ROOT=Path(__file__).resolve().parents[2];D=ROOT/'outputs/universal_tracker_20261006';out=D/'figures/long_horizon';out.mkdir(exist_ok=True)
plt.rcParams.update({'font.size':10,'svg.fonttype':'none','pdf.fonttype':42});fig,axes=plt.subplots(1,3,figsize=(14,4.6));colours=['#dc8041','#198db2'];records=[]
for i,(name,title) in enumerate([('frozen_demo_long_horizon_v3','Relative command anchors'),('frozen_demo_long_no_anchors','Fixed world reference')]):
 run=D/'general_evaluation'/name/'four_cycles';q=np.load(run/'actual.npz')['qpos'];ref=np.load(run/'initial_motion.npz')['body_pos_w'][:,0];n=len(q);error=np.linalg.norm(q[:,:2]-ref[:n,:2],axis=-1);ax=axes[i];ax.plot(ref[:,0],ref[:,1],'--',color='#8c99a4',lw=1.4,label='Initial world plan');ax.plot(q[:,0],q[:,1],color=colours[i],lw=1.4,label='Actual physical root');ax.scatter(q[0,0],q[0,1],marker='o',color='black',s=25,label='Start');ax.scatter(q[-1,0],q[-1,1],marker='x',color=colours[i],s=45,label='End');ax.set(title=title,xlabel='World x (m)',ylabel='World y (m)');ax.axis('equal');ax.grid(alpha=.2);ax.legend(fontsize=8);axes[2].plot(np.arange(n)/50,error,color=colours[i],lw=1,label=title);records.append(dict(mode=title,frames=n,final_world_plan_root_xy_error_m=float(error[-1])))
axes[2].set(title='Deviation from the INITIAL world plan',xlabel='Time (s)',ylabel='Root XY deviation (m)');axes[2].legend(fontsize=8);axes[2].grid(alpha=.2);fig.suptitle('Same frozen tracker, same six generated clips repeated four times',fontsize=13);fig.text(.5,.018,'One rollout per mode; 24 stages are not independent trials. Anchoring intentionally moves future references. No physical-state resets.',ha='center',fontsize=9,color='#536675');fig.tight_layout(rect=[0,.065,1,.93])
for ext in ['png','pdf','svg']:fig.savefig(out/('long_horizon.'+ext),dpi=180,bbox_inches='tight')
(out/'protocol.json').write_text(json.dumps(dict(scope=__doc__,records=records),indent=2));print(out)
