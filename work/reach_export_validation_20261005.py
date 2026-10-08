import sys,json,argparse
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/reach_demo_20261005';sys.path.insert(0,str(R/'work'));import reach_adapter_20261005 as ra
from core import torch,np,FK
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--step',type=int,required=True);a=p.parse_args();folder=D/a.name;out=folder/f'export_{a.step:05d}';out.mkdir(exist_ok=False);z=np.load(folder/f'validation_{a.step:05d}.npz');raw=torch.from_numpy(z['motion']);fk=FK();pos,poses,root=fk(raw,canonical=False,return_pose=True);rows=[]
for i,c in enumerate(z['commands']):
 path=out/f'reach_c{i}.npz';np.savez_compressed(path,motion=raw[i].numpy(),joints_zup_m=pos[i].numpy(),poses_axisangle=poses[i].numpy(),root_translation=root[i].numpy(),human_height=fk.height,fps=20.,task='reach',command=float(c));rows.append(dict(index=i,command=float(c),path=str(path)))
(out/'manifest.json').write_text(json.dumps(rows,indent=2));print(out)
