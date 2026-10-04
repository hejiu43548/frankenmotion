"""Scientific skeleton replay: native generated human, G1 reference, simulated G1."""
import sys,os,json
from pathlib import Path
os.environ['MPLBACKEND']='Agg'
BASE=Path('/home/pku/frankenmotion/outputs_amass/franken_eleven_20261003');OUT=BASE.parent/'franken_improve_20261003';sys.path.insert(0,str(BASE/'code'))
import transfer as tr
import numpy as np
import matplotlib.pyplot as plt
import imageio.v2 as imageio
m=tr.rt.load_model();parents=np.load(BASE/'skeleton.npz')['parents'][:22];links=[(int(parents[j]),j) for j in range(1,22)]
robotlinks=[(0,1),(1,4),(4,7),(0,2),(2,5),(5,8),(0,16),(0,17),(16,17),(16,18),(18,20),(17,19),(19,21)]
folder=OUT/'visuals';folder.mkdir(exist_ok=True)
for task in ['wave','lean','back_walk','jump']:
 name=f'{task}_p0_s0_c4';path=OUT/'physical_development_eval/gmr_probe'/(name+'_stock.npz')
 if not path.exists():continue
 z=np.load(OUT/'generated/physical_development'/(name+'.npz'));human=z['joints_zup_m'].astype(float);side=human[0,1]-human[0,2];yaw=np.arctan2(side[1],side[0])-np.pi/2;human=tr.Rotation.from_euler('z',-yaw).apply(human.reshape(-1,3)).reshape(human.shape);human[:,:,:2]-=human[0,0,:2].copy();a=np.load(path);reference=tr.get_positions(m,a['reference_qpos']);actual=tr.get_positions(m,a['qpos']);ts=a['time_s'];want=1+np.arange(len(human))*.05
 actual=np.stack([np.interp(want,ts,x) for x in actual.reshape(len(actual),-1).T],1).reshape(len(human),24,3)
 # Maintain a shared world-scale view and show only actually observed samples.
 n=sum(want<=ts[-1]+1e-7);human=human[:n];reference=reference[:n];actual=actual[:n]
 fig=plt.figure(figsize=(12,4),dpi=90);axes=[fig.add_subplot(1,3,i+1,projection='3d') for i in range(3)];seqs=[human,reference,actual];colors=['#64748b','#168ab0','#e47736'];names=['Generated human','GMR G1 reference','SONIC v1.1 physical']
 artists=[]
 for ax,p,col,title in zip(axes,seqs,colors,names):
  center=p[:,:,0:2].reshape(-1,2).mean(0);span=max(1.3,np.ptp(p[:,:,[0,1]],axis=(0,1)).max()/2+.25)
  ax.set_xlim(center[0]-span,center[0]+span);ax.set_ylim(center[1]-span,center[1]+span);ax.set_zlim(0,2);ax.view_init(elev=15,azim=-65);ax.set_box_aspect((2*span,2*span,2));ax.set_title(title,fontsize=10);ax.set_xlabel('X (m)');ax.set_ylabel('Y (m)')
  segs=links if p is human else robotlinks;arts=[ax.plot([],[],[],color=col,lw=2)[0] for _ in segs];artists.append((arts,segs))
 title=fig.suptitle(task+' | high command | development only');fig.tight_layout()
 with imageio.get_writer(folder/(task+'_development.mp4'),fps=20,codec='libx264',quality=7) as writer:
  for i in range(n):
   for p,(arts,segs) in zip(seqs,artists):
    for art,(aa,bb) in zip(arts,segs):
     q=p[i,[aa,bb]];art.set_data(q[:,0],q[:,1]);art.set_3d_properties(q[:,2])
   title.set_text(f'{task} | high command | t={i/20:.2f}s | development only');fig.canvas.draw();writer.append_data(np.asarray(fig.canvas.buffer_rgba())[:,:,:3])
   if i==n//2:fig.savefig(folder/(task+'_midpoint.png'))
 plt.close(fig);print(task,n,flush=True)
