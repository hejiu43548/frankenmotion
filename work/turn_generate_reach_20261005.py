import sys,json,argparse,hashlib
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/turn_demo_20261005';sys.path.insert(0,str(R/'work'));import reach_adapter_v6_20261005 as ra
from core import torch,np,FK,sample
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--walk-folder',required=True);p.add_argument('--seed',type=int,default=92005000);a=p.parse_args();out=D/a.name;out.mkdir(exist_ok=False);weight=R/'outputs_amass/reach_demo_20261005/frozen/reach_adapter.pt';m=ra.load(weight);torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.12);fk=FK('cuda');base=json.loads((Path(a.walk_folder)/'manifest.json').read_text());rows=[]
for i in range(len(base)*2):
 c=[.3,.5][i%2];seed=a.seed+i//2;dest=out/f'scene_{i:03d}';dest.mkdir();local,tx,cmd,controls,cond=ra.setup([c],prompt=0);m.denoiser.command=cond;raw=sample(m,local,tx,1,cmd,[seed],True,steps=50)
 with torch.no_grad():pos,poses,root=fk(raw,canonical=False,return_pose=True)
 np.savez_compressed(dest/'human_reach.npz',motion=raw[0].cpu().numpy(),joints_zup_m=pos[0].cpu().numpy(),poses_axisangle=poses[0].cpu().numpy(),root_translation=root[0].cpu().numpy(),human_height=fk.height,fps=20.,task='reach',command=c);rows.append(dict(index=i,source=str(dest),command=c,exit_task='none',exit_command=0.,seed=seed,walk_source=base[i//2]['source'],generator_weight=str(weight),generator_sha256=hashlib.sha256(weight.read_bytes()).hexdigest()));print(i,c,flush=True)
(out/'manifest.json').write_text(json.dumps(rows,indent=2))
