import argparse,json,numpy as np,mujoco
from pathlib import Path
from scipy.signal import find_peaks
p=argparse.ArgumentParser();p.add_argument('--scene',required=True);p.add_argument('--run',required=True);a=p.parse_args();s=Path(a.scene);meta=json.loads((s/'reference_contact.json').read_text());lo,hi=meta['segments']['exit'];q=np.load(s/'reference_contact.npz')['reference_qpos'][lo:hi];m=mujoco.MjModel.from_binary_path(str(s/a.run/'scene.mjb'));d=mujoco.MjData(m);floor=m.geom('terrain').id;gids=[[i for i in range(m.ngeom) if m.geom(i).name.startswith('robot/'+side+'_foot') and 'collision' in m.geom(i).name] for side in ['left','right']];zs=[]
for state in q:
 d.qpos[:]=state;mujoco.mj_forward(m,d);zs.append([min(mujoco.mj_geomDistance(m,d,g,floor,1.,None) for g in foot) for foot in gids])
z=np.array(zs);print(json.dumps(dict(quantiles=np.quantile(z,[0,.5,.95,1],axis=0).tolist(),peaks=[find_peaks(z[:,i],height=.015,prominence=.012,distance=5)[0].tolist() for i in range(2)],both_below_5mm=float((z<.005).all(1).mean())),indent=2))
