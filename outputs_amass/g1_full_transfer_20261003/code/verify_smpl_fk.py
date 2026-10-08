import os,sys,json
from pathlib import Path
import numpy as np
import torch
ROOT=Path('/home/pku/frankenmotion/work/g1_sonic_official');OUT=Path(os.environ['FULL_OUT'])
sys.path.insert(0,str(ROOT))
from gear_sonic.trl.utils.torch_transform import compute_human_joints
from scipy.spatial.transform import Rotation
z=np.load(OUT/'human/01_walk_speed_v1.npz');poses=z['poses_axisangle'][[0,50,119]]
expected=compute_human_joints(torch.from_numpy(poses[:,3:]),torch.from_numpy(poses[:,:3]),human_joints_info_path=str(OUT/'human_joints_info.pkl')).numpy()
asset=np.load(OUT/'official_human_skeleton.npz');J=asset['J'].reshape(-1,3);parents=asset['parents'].reshape(-1)
angles=np.pad(poses,((0,0),(0,99))).reshape(-1,55,3);R=Rotation.from_rotvec(angles.reshape(-1,3)).as_matrix().reshape(-1,55,3,3)
globalR=[];positions=[]
for i in range(55):
 if i==0:globalR.append(R[:,i]);positions.append(np.repeat(J[i][None],len(poses),axis=0))
 else:
  p=int(parents[i]);globalR.append(globalR[p]@R[:,i]);positions.append(positions[p]+np.einsum('nij,j->ni',globalR[p],J[i]-J[p]))
actual=np.stack(positions,axis=1)[:,list(range(22))+[39,54]]
error=float(np.max(np.abs(expected-actual)));assert error<2e-6,error
(OUT/'smpl_fk_parity.json').write_text(json.dumps(dict(max_error_m=error,frames=[0,50,119],official_function='compute_human_joints',passed=True),indent=2));print(error)
