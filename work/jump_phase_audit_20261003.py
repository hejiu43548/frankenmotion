"""Compare jump timing/relative leg extension for development data only."""
import sys,os,json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
BASE=Path('/home/pku/frankenmotion/outputs_amass/franken_eleven_20261003');OUT=BASE.parent/'franken_improve_20261003';sys.path.insert(0,str(BASE/'code'));import transfer as tr
m=tr.rt.load_model();fig,axes=plt.subplots(5,3,figsize=(13,14));rows=[]
for ci,cmd in enumerate([.25,.325,.4,.475,.55]):
 name=f'jump_p0_s0_c{ci}';z=np.load(OUT/'generated/jump_aligned_development'/(name+'.npz'));v=np.load(OUT/'jump_aligned_development_eval/gmr_probe'/(name+'_uniform.npz'));human=z['joints_zup_m'];g1=tr.get_positions(m,v['reference_qpos']);actual50=tr.get_positions(m,v['qpos']);want=1+np.arange(60)*.05;actual=np.stack([np.interp(want,v['time_s'],x) for x in actual50.reshape(len(actual50),-1).T],1).reshape(60,24,3)
 for col,(label,p,height) in enumerate([('Generated human',human,float(z['human_height'])),('GMR uniform reference',g1,tr.robot_height(m)),('SONIC v1.1 actual',actual,tr.robot_height(m))]):
  scale=tr.HH/height;t=np.arange(60)/20.;root=(p[:,0,2]-p[0,0,2])*scale;feet=p[:,[7,8],2];floor=feet[0].min();clearance=(feet.min(1)-floor)*scale;air=clearance>.08;edges=np.diff(np.r_[False,air,False].astype(int));segments=list(zip(np.where(edges==1)[0],np.where(edges==-1)[0]));relrange=np.ptp(feet-p[:,0,2,None],axis=0)*scale
  ax=axes[ci,col];ax.plot(t,root,label='Pelvis rise',color='#188db6');ax.plot(t,clearance,label='Lower ankle rise',color='#d58436');ax.axhline(cmd,color='#999',ls='--',lw=.8,label='Command')
  for a,b in segments:ax.axvspan(a/20,min(b/20,t[-1]),alpha=.08,color='#36976a')
  ax.set_title(f'{label} | command {cmd:.3f} m',fontsize=9);ax.set_xlabel('Time (s)');ax.set_ylabel('Human-equivalent m');ax.grid(alpha=.12);ax.set_ylim(-.25,.7)
  rows.append(dict(command=cmd,stage=label,air_segments_frames=segments,ankle_relative_pelvis_z_range_m=relrange.tolist(),metrics=tr.measure(p,'jump',height)))
axes[0,0].legend(fontsize=7);fig.suptitle('Jump development timing: aligned-teacher generator (5 commands, one source)',fontsize=14);fig.tight_layout(rect=[0,0,1,.97]);folder=OUT/'visuals_jump_aligned';folder.mkdir(exist_ok=True);fig.savefig(folder/'jump_phase_audit.png',dpi=150);(folder/'jump_phase_audit.json').write_text(json.dumps(rows,indent=2,default=lambda x:int(x) if isinstance(x,np.integer) else x.item()))
