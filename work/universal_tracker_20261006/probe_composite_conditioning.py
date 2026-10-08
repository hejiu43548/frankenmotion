"""Generation-only conditioning ablation; no pose edits, no tracker selection."""
import sys,json,hashlib
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/universal_tracker_20261006';sys.path[:0]=[str(R/'work'),str(R/'outputs_amass/franken_eleven_20261003/code')]
import physical_adapter_20261003
from core import torch,np,FK,load_model,sample,encode_control,HH
out=D/'composite_conditioning_probe';out.mkdir(exist_ok=False);torch.set_num_threads(4);weight=R/'outputs_amass/franken_improve_20261003/frozen_generation_v1/physical.pt';model,_=load_model('cpu',weight);fk=FK('cpu');prompt=D/'generated_demo_candidates/prompts/walk_wave.json';z=torch.load(prompt.with_suffix('.pt'),map_location='cpu',weights_only=False);rows=[]
for mode in ['task_walk','root_only','task_wave']:
 for si in range(2):
  for speed,amplitude in [(s,a) for s in [.35,.55] for a in [.12,.2]]:
   seed=108061000+si;local=z['local'][None];tx=z['tx'];values=torch.zeros(1,120,2);values[...,0]=speed*fk.height/HH;controls=encode_control(values,torch.ones(1,120,dtype=torch.bool));command=torch.tensor([amplitude if mode=='task_wave' else speed]);raw=sample(model,local,tx,3 if mode=='task_wave' else 10,command,[seed],mode!='root_only',root_controls=controls)
   with torch.no_grad():pos,poses,root=fk(raw,canonical=False,return_pose=True)
   path=out/f'{mode}_s{si}_v{speed}_a{amplitude}.npz';np.savez_compressed(path,motion=raw[0].numpy(),joints_zup_m=pos[0].numpy(),poses_axisangle=poses[0].numpy(),root_translation=root[0].numpy(),human_height=fk.height,fps=20.);rows.append(dict(task='walk_wave',mode=mode,source=f'{mode}_s{si}',seed=seed,command=speed,wave_command=amplitude if mode=='task_wave' else None,wave_grid_label=amplitude,path=str(path),source_type='FrankenMotion_generated_multicondition_probe',split='development',frames=120,generator_sha256=hashlib.sha256(weight.read_bytes()).hexdigest()));(out/'manifest.json').write_text(json.dumps(rows,indent=2));print(mode,si,speed,amplitude,flush=True)
(out/'protocol.json').write_text(json.dumps(dict(scope=__doc__,mode_description={'task_walk':'NumericTaskControl walk plusRootControl speed; amplitudegridisduplicatecontrol','root_only':'RootControl speed plusparttext,noTaskControl; amplitudegridisduplicatecontrol','task_wave':'TaskControl wave amplitude plusRootControl speed simultaneously'},samples=24),indent=2))
