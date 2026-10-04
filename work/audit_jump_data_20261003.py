"""Inspect cached TRAIN jump exemplars; preserve family-disjoint validation data."""
import os,sys,json
from pathlib import Path
BASE=Path('/home/pku/frankenmotion/outputs_amass/franken_eleven_20261003');OUT=BASE.parent/'franken_improve_20261003';os.environ['ELEVEN_OUT']=str(BASE);sys.path.insert(0,str(BASE/'code'))
from core import torch,np,FK
from partner_metrics import measure
NAMES=['Pelvis','L_Hip','R_Hip','Torso1','L_Knee','R_Knee','Torso2','L_Ankle','R_Ankle','Torso','L_Toe','R_Toe','Neck','L_Collar','R_Collar','Head','L_Shoulder','R_Shoulder','L_Elbow','R_Elbow','L_Wrist','R_Wrist','L_Hand','R_Hand']
torch.set_num_threads(2);fk=FK('cpu');data=json.loads((BASE/'data_manifest.json').read_text());rows=[];folder=OUT/'jump_training_exemplars';folder.mkdir(exist_ok=True)
for r in data:
 if r['task']!='jump' or r['split']!='train':continue
 x=torch.load(r['path'],map_location='cpu',weights_only=False)['x'][None,:,:205]
 with torch.no_grad():p,poses,root=fk(x,canonical=False,return_pose=True)
 p=p[0].numpy();metric=measure('jump',p,NAMES,distance_ratio=fk.height/1.2701193988323212);metric['quantity']*=1.2701193988323212/fk.height;ank=p[:,[7,8],2];floor=np.min(ank[0]);air=ank.min(1)>floor+.08;z=p[:,0,2];az=np.diff(z,n=2)*400;mask=air[1:-1];ballistic=float(np.mean(abs(az[mask]+9.81))) if mask.any() else None
 row=dict(r,metrics=metric,air_frames=int(air.sum()),ballistic_mae=ballistic,knee_range=float(np.ptp(p[:,4,2]-p[:,0,2])));rows.append(row)
 np.savez_compressed(folder/(r['key']+'.npz'),joints_zup_m=p,poses_axisangle=poses[0].numpy(),root_translation=root[0].numpy(),human_height=fk.height,fps=20.)
rows.sort(key=lambda r:(not r['metrics']['event_pass'],r['ballistic_mae'] if r['ballistic_mae'] is not None else 100))
(folder/'audit.json').write_text(json.dumps(rows,indent=2,default=lambda x:x.item()));print(json.dumps(rows[:8],indent=2,default=lambda x:x.item()))
