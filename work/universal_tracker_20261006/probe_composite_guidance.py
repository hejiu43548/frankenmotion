"""Exploratory latent command-guidance scaling, not new training or pose editing."""
import sys,json,hashlib
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/universal_tracker_20261006';sys.path[:0]=[str(R/'work'),str(R/'outputs_amass/franken_eleven_20261003/code')]
import physical_adapter_20261003 as pa
class Guided(pa.TemporalTaskControl):
 gain=1.
 def hook(self,i):
  original=super().hook(i)
  def apply(module,args,output):return output+self.gain*(original(module,args,output)-output)
  return apply
pa.core.TaskControl=Guided
from core import torch,np,FK,load_model,sample,encode_control,HH
out=D/'composite_guidance_probe';out.mkdir(exist_ok=False);torch.set_num_threads(4);weight=R/'outputs_amass/franken_improve_20261003/frozen_generation_v1/physical.pt';model,_=load_model('cpu',weight);fk=FK('cpu');prompt=D/'generated_demo_candidates/prompts/walk_wave.json';z=torch.load(prompt.with_suffix('.pt'),map_location='cpu',weights_only=False);rows=[]
for gain,speed in [(g,s) for g in [.1,.25,.5] for s in [.5,1.,1.5]]+[(0.,.55),(1.,.55)]:
 model.denoiser.gain=gain;seed=108061001;values=torch.zeros(1,120,2);values[...,0]=speed*fk.height/HH;controls=encode_control(values,torch.ones(1,120,dtype=torch.bool));raw=sample(model,z['local'][None],z['tx'],3,torch.tensor([.2]),[seed],True,root_controls=controls)
 with torch.no_grad():pos,poses,root=fk(raw,canonical=False,return_pose=True)
 path=out/f'g{gain}_v{speed}.npz';np.savez_compressed(path,motion=raw[0].numpy(),joints_zup_m=pos[0].numpy(),poses_axisangle=poses[0].numpy(),root_translation=root[0].numpy(),human_height=fk.height,fps=20.);rows.append(dict(task='walk_wave',mode='latent_gain',gain=gain,source=f'g{gain}',seed=seed,command=speed,wave_command=.2,path=str(path),source_type='FrankenMotion_generated_latent_guidance',split='development',frames=120));(out/'manifest.json').write_text(json.dumps(rows,indent=2));print(gain,speed,flush=True)
 if gain in [0.,1.]:
  previous=D/'composite_conditioning_probe'/f'{"root_only" if gain==0 else "task_wave"}_s1_v0.55_a0.2.npz';error=float(np.max(abs(raw[0].numpy()-np.load(previous)['motion'])));assert error<1e-5,error;print('endpoint_parity',gain,error,flush=True)
