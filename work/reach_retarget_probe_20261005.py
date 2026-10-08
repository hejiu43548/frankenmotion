import sys,json,numpy as np
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/reach_demo_20261005';sys.path.insert(0,str(R/'work'));import unified_retarget_20261004 as rt
rt.init();m=rt.g.M;d=rt.g.tr.mujoco.MjData(m);mj=rt.g.tr.mujoco;hand=m.joint('right_wrist_yaw_joint').bodyid[0];rows=[]
for row in json.loads((D/'command_probe/manifest.json').read_text()):
 q=rt.g.convert(np.load(row['path']),'uniform');p=[];w=[];rot=[]
 for state in q:
  d.qpos[:]=state;mj.mj_forward(m,d);p.append(d.xpos[hand]+d.xmat[hand].reshape(3,3)@np.array([.11,.01,0.]));w.append(d.xpos[hand].copy());rot.append(d.xmat[hand].reshape(3,3).copy())
 p=np.array(p);w=np.array(w);np.savez_compressed(Path(row['path']).with_name(row['name']+'_g1.npz'),reference_qpos=q,palm=p,wrist=w,hand_rotation=rot)
 row.update(g1_palm_x_p95=float(np.quantile((p-q[:,:3])[:,0],.95)),g1_palm_last=p[-1].tolist(),g1_palm_z_range=[float(p[:,2].min()),float(p[:,2].max())],g1_palm_peak=p[int(np.argmax((p-q[:,:3])[:,0]))].tolist(),g1_hand_rel_last=(p[-1]-q[-1,:3]).tolist());rows.append(row);print(row['name'],row['g1_palm_peak'],row['g1_palm_last'],flush=True)
(D/'command_probe/retarget_metrics.json').write_text(json.dumps(rows,indent=2))
