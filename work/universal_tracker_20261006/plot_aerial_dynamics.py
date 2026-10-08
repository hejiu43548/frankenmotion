from pathlib import Path
import json
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
R=Path(__file__).resolve().parents[2];D=R/'outputs/universal_tracker_20261006';out=D/'figures/aerial_dynamics';out.mkdir(parents=True,exist_ok=True)
rows=json.loads((D/'aerial_dynamics_v4.json').read_text())['results'];fig,axs=plt.subplots(2,2,figsize=(10.4,6.3),sharex='col')
for col,cmd in enumerate([.325,.55]):
 row=next(r for r in rows if r['source']=='jump_p0_s0' and abs(r['command']-cmd)<1e-5)
 for key,color,label in [('reference','#d56b35','G1 reference'),('actual','#1677a4','Physical rollout')]:
  s=row[key]['series'];t=np.arange(len(s['com']))*.02;com=np.asarray(s['com']);acc=np.asarray(s['acceleration']);mask=np.asarray(s['interior_air']);axs[0,col].plot(t,com[:,2],color=color,label=label,lw=2);y=np.where(mask,acc[:,2],np.nan);axs[1,col].plot(t,y,color=color,lw=2,label=label)
 axs[0,col].set_title(f'Jump command {cmd:.3f} m (human-equivalent)');axs[0,col].set_ylabel('Whole-robot COM height (m)');axs[1,col].axhline(-9.81,color='#505d66',ls='--',lw=1.4,label='Gravity');axs[1,col].set_ylabel('COM vertical acceleration (m/s²)');axs[1,col].set_xlabel('Time after standing entry (s)')
 for ax in axs[:,col]:ax.grid(alpha=.18);ax.spines[['top','right']].set_visible(False)
 axs[1,col].set_ylim(-40,40)
axs[0,0].legend(frameon=False,loc='upper left',fontsize=9);axs[1,0].legend(frameon=False,loc='upper left',fontsize=9)
fig.suptitle('Reference physics diagnostic: airborne COM must accelerate under gravity',fontsize=13,y=.99)
fig.text(.5,.015,'Development source jump_p0_s0. Acceleration shown only when all 11 filter frames are contact-free with feet >2 cm above ground.\nCubic 0.22 s derivative window; reference includes hovering preparation/landing. This is not a full feasibility certificate.',ha='center',fontsize=8.5,color='#48565f')
fig.tight_layout(rect=[0,.09,1,.96])
for ext in ['png','pdf','svg']:fig.savefig(out/('aerial_dynamics.'+ext),dpi=190,bbox_inches='tight')
print(out)
