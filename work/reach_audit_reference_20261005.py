import json,sys,argparse,numpy as np
from pathlib import Path
from scipy.spatial.transform import Rotation
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/reach_demo_20261005';sys.path.insert(0,str(R/'work'));import unified_retarget_20261004 as rt
p=argparse.ArgumentParser();p.add_argument('--folder',required=True);a=p.parse_args();folder=Path(a.folder);rt.init();m=rt.g.M;d=rt.g.tr.mujoco.MjData(m);wrist=m.joint('right_wrist_yaw_joint').id;rows=[]
for row in json.loads((folder/'manifest.json').read_text()):
 s=Path(row['source']);z=np.load(s/'human_reach.npz');hp=z['joints_zup_m'];side=hp[0,1,:2]-hp[0,2,:2];angle=np.arctan2(side[1],side[0])-np.pi/2;f=np.array([np.cos(angle),np.sin(angle),0]);hq=((hp[48:72,21]-hp[48:72,0])@f)*1.2701193988323212/float(z['human_height']);q=np.load(s/'reference_contact.npz')['reference_qpos'];meta=json.loads((s/'reference_contact.json').read_text());lo,hi=meta['segments']['contact_hold'];f=np.array([np.cos(row['direction_rad']),np.sin(row['direction_rad']),0]);rq=[];palms=[]
 for state in q[lo:hi]:
  d.qpos[:]=state;rt.g.tr.mujoco.mj_forward(m,d);rq.append(float((d.xanchor[wrist]-d.xpos[m.body('pelvis').id])@f)*1.2701193988323212/1.0486437524221748);hand=m.joint('right_wrist_yaw_joint').bodyid[0];palms.append(d.xpos[hand]+d.xmat[hand].reshape(3,3)@np.array([.11,.01,0]))
 rows.append(dict(index=row['index'],command=row['reach_command_human_m'],exit_task=row['exit_task'],human_hold_mean=float(np.mean(hq)),human_hold_std=float(np.std(hq)),g1_reference_hold_mean=float(np.mean(rq)),g1_reference_hold_std=float(np.std(rq)),g1_palm_mean=np.mean(palms,0).tolist(),source=str(s)))
(folder/'command_reference_audit.json').write_text(json.dumps(rows,indent=2));print(json.dumps(rows,indent=2))
