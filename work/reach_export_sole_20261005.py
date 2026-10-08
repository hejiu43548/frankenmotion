import json,mujoco
from pathlib import Path
from scipy.spatial.transform import Rotation
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/reach_demo_20261005';m=mujoco.MjModel.from_binary_path(str(D/'development_v8/scene_002/tracker_v5_1999_valid/scene.mjb'));out={}
for side in ['left','right']:
 ps=[];rs=[]
 for g in range(m.ngeom):
  if m.geom(g).name.startswith('robot/'+side+'_foot') and 'collision' in m.geom(g).name:
   assert int(m.geom_type[g])==3
   axis=Rotation.from_quat(m.geom_quat[g][[1,2,3,0]]).as_matrix()[:,2]*m.geom_size[g,1];ps.extend([(m.geom_pos[g]+axis).tolist(),(m.geom_pos[g]-axis).tolist()]);rs.extend([float(m.geom_size[g,0])]*2)
 out[side]={'points':ps,'radii':rs}
(D/'sole_geometry.json').write_text(json.dumps(out,indent=2))
