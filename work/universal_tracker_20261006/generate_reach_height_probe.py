"""CPU generation-only height-conditioning sweep, without output pose edits."""
import sys,json,hashlib
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/universal_tracker_20261006';sys.path.insert(0,str(R/'work'))
import reach_adapter_v6_20261005 as ra
from core import torch,np,FK,sample
from generate import inputs
torch.set_num_threads(4);out=D/'reach_height_probe';out.mkdir(exist_ok=False)
m,_=ra.ga.pa.core.load_model('cpu',ra.ga.BASE);m.denoiser=ra.ga.GoalControl(m.denoiser)
goal=D/'backup/stable_frozen/goal_adapter.pt';state=torch.load(goal,map_location='cpu',weights_only=False);result=m.denoiser.load_state_dict(state['adapter'],strict=False);assert not result.unexpected_keys and all(k.startswith('base.') for k in result.missing_keys)
m.denoiser=ra.ReachControl(m.denoiser);weight=D/'backup/stable_frozen/reach_adapter.pt';state=torch.load(weight,map_location='cpu',weights_only=False);result=m.denoiser.load_state_dict(state['adapter'],strict=False);assert not result.unexpected_keys and all(k.startswith('base.') for k in result.missing_keys);m.eval();fk=FK('cpu');rows=[]
for seed in [86005000,86005001]:
 for command in [.3,.5]:
  for height in [.84,.86,.88,.90]:
   src=dict(prompt=str(ra.D/'prompts/reach_place_p0.json'),task='reach',frames=120);local,tx,cmd,_=inputs(src,[command],device='cpu');m.denoiser.command=torch.tensor([[command,height]],dtype=torch.float32);raw=sample(m,local,tx,1,cmd,[seed],True,steps=50)
   with torch.no_grad():pos,poses,root=fk(raw,canonical=False,return_pose=True)
   p=out/f'reach_{seed}_{command:.2f}_{height:.2f}.npz';np.savez_compressed(p,motion=raw[0].numpy(),joints_zup_m=pos[0].numpy(),poses_axisangle=poses[0].numpy(),root_translation=root[0].numpy(),human_height=fk.height,fps=20.)
   rows.append(dict(task='reach',command=command,height_command_robot_m=height,seed=seed,path=str(p),generator_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),human_sha256=hashlib.sha256(p.read_bytes()).hexdigest(),scope='Existing learned height input sweep; .88 and .90 exceed training range .81-.87 and are explicitly exploratory. No pose correction.'))
   (out/'manifest.json').write_text(json.dumps(rows,indent=2));print(seed,command,height,flush=True)
