"""Inject user reach/departure commands into motion generation; never edit output hand poses."""
import argparse,sys,json,hashlib
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/reach_demo_20261005';sys.path.insert(0,str(R/'work'));import reach_adapter_v6_20261005 as ra
from core import torch,np,FK,sample
from generate import inputs
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--walk-source',required=True);p.add_argument('--weight',default=str(D/'frozen/reach_adapter.pt'));p.add_argument('--reach',type=float,required=True);p.add_argument('--exit-task',choices=['back_walk','sidestep','none'],default='back_walk');p.add_argument('--exit-command',type=float);p.add_argument('--seed',type=int,default=88005000);a=p.parse_args();assert .28<=a.reach<=.52;out=D/a.name;out.mkdir(exist_ok=False);s=out/'scene_000';s.mkdir();torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.14);m=ra.load(a.weight);fk=FK('cuda');local,tx,cmd,controls,cond=ra.setup([a.reach]);m.denoiser.command=cond;raw=sample(m,local,tx,1,cmd,[a.seed],True,steps=50)
def save(raw,name,task,command):
 with torch.no_grad():
  pos,poses,root=fk(raw,canonical=False,return_pose=True);np.savez_compressed(s/name,motion=raw[0].cpu().numpy(),joints_zup_m=pos[0].cpu().numpy(),poses_axisangle=poses[0].cpu().numpy(),root_translation=root[0].cpu().numpy(),human_height=fk.height,fps=20.,task=task,command=command)
save(raw,'human_reach.npz','reach',a.reach);exit_cmd=0.
if a.exit_task!='none':
 exit_cmd=a.exit_command if a.exit_command is not None else .35 if a.exit_task=='back_walk' else .5
 assert (.3<=exit_cmd<=.6 if a.exit_task=='back_walk' else .4<=exit_cmd<=.7)
 sources=json.loads((ra.ga.pa.BASE/'evaluation_manifest.json').read_text());src=next(r for r in sources if r['source']==a.exit_task+'_p0_s0');local,tx,cmd,controls=inputs(src,[exit_cmd]);m.denoiser.command=None;raw=sample(m,local,tx,src['task_id'],cmd,[a.seed],True,root_controls=controls);save(raw,'human_exit.npz',a.exit_task,exit_cmd)
row=dict(index=0,source=str(s),command=a.reach,exit_task=a.exit_task,exit_command=exit_cmd,seed=a.seed,walk_source=str(Path(a.walk_source).resolve()),generator_weight=str(Path(a.weight).resolve()),generator_sha256=hashlib.sha256(Path(a.weight).read_bytes()).hexdigest());(out/'manifest.json').write_text(json.dumps([row],indent=2));print(out)
