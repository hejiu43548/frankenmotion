import sys,json,argparse,hashlib
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/reach_demo_20261005';sys.path.insert(0,str(R/'work'));import reach_adapter_v6_20261005 as ra;import reach_exit_adapter_v2_20261005 as ea
from core import torch,np,FK,sample
from generate import inputs
p=argparse.ArgumentParser();p.add_argument('--weight',required=True);p.add_argument('--exit-weight',required=True);p.add_argument('--name',required=True);p.add_argument('--count',type=int,default=48);p.add_argument('--seed',type=int,default=84005000);p.add_argument('--dev',action='store_true');a=p.parse_args();out=D/a.name;out.mkdir(exist_ok=False);torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.14);m=ea.load(a.exit_weight,reach_weight=a.weight);fk=FK('cuda');rng=np.random.default_rng(a.seed);base=json.loads((R/'outputs_amass/table_demo_20261005/training_v2/manifest.json').read_text());sources=json.loads((ra.ga.pa.BASE/'evaluation_manifest.json').read_text());rows=[]
for i in range(a.count):
 c=float([.3,.5][i%2] if a.dev else rng.uniform(.28,.52));exit_task=['none','back_walk','sidestep'][(i//2)%3 if a.dev else i%3];seed=a.seed+(i//2 if a.dev else i);dest=out/f'scene_{i:03d}';dest.mkdir();local,tx,cmd,controls,cond=ra.setup([c],prompt=0 if a.dev else i%2);m.denoiser.condition=None;m.denoiser.base.command=cond;raw=sample(m,local,tx,1,cmd,[seed],True,steps=50)
 def save(raw,name,task,command):
  pos,poses,root=fk(raw,canonical=False,return_pose=True);np.savez_compressed(dest/name,motion=raw[0].cpu().numpy(),joints_zup_m=pos[0].cpu().numpy(),poses_axisangle=poses[0].cpu().numpy(),root_translation=root[0].cpu().numpy(),human_height=fk.height,fps=20.,task=task,command=command)
 with torch.no_grad():save(raw,'human_reach.npz','reach',c)
 exit_cmd=0.
 if exit_task!='none':
  src=next(r for r in sources if r['source']==f'{exit_task}_p{0 if a.dev else i%2}_s0');exit_cmd=float((.35 if exit_task=='back_walk' else .5) if a.dev else (rng.uniform(.32,.4) if exit_task=='back_walk' else rng.uniform(.4,.7)));local,tx,cmd,exitcond,tid=ea.setup(exit_task,[exit_cmd]);m.denoiser.base.command=None;m.denoiser.condition=exitcond;raw=sample(m,local,tx,tid,cmd,[seed],False,steps=50)
  with torch.no_grad():save(raw,'human_exit.npz',exit_task,exit_cmd)
 row=dict(index=i,source=str(dest),command=c,exit_task=exit_task,exit_command=exit_cmd,seed=seed,walk_source=str(R/f'outputs_amass/gait_demo_20261005/dev_timed/scene_{(i//2)%6:03d}') if a.dev else base[i%len(base)]['source'],exit_generator_weight=str(Path(a.exit_weight).resolve()),exit_generator_sha256=hashlib.sha256(Path(a.exit_weight).read_bytes()).hexdigest(),generator_weight=str(Path(a.weight).resolve()),generator_sha256=hashlib.sha256(Path(a.weight).read_bytes()).hexdigest());rows.append(row);print(i,c,exit_task,flush=True)
(out/'manifest.json').write_text(json.dumps(rows,indent=2))
