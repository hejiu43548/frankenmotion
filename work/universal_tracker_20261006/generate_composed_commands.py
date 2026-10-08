"""Test existing shared task-command adapter on new part-wise composite prompts."""
import sys,json,hashlib,time
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/universal_tracker_20261006';sys.path.insert(0,str(R/'outputs_amass/franken_eleven_20261003/code'))
sys.path.insert(0,str(R/'work'))
import physical_adapter_20261003
from core import torch,np,load_model,FK,sample,encode_control
out=D/'generated_composed_commands';out.mkdir(exist_ok=False);(out/'human').mkdir()
torch.set_num_threads(4);weight=R/'outputs_amass/franken_improve_20261003/frozen_generation_v1/physical.pt';model,_=load_model('cpu',weight);fk=FK('cpu');rows=[]
for task in ['walk_wave','walk_point']:
 prompt=D/'generated_demo_candidates/prompts'/(task+'.json');z=torch.load(prompt.with_suffix('.pt'),map_location='cpu',weights_only=False);commands=[.35,.5,.55,.8];b=len(commands);local=z['local'][None].expand(b,-1,-1);tx={k:v.repeat((b,)+(1,)*(v.ndim-1)) if torch.is_tensor(v) else v for k,v in z['tx'].items()};cmd=torch.tensor(commands);values=torch.zeros(b,120,2);values[...,0]=cmd[:,None];controls=encode_control(values,torch.ones(b,120,dtype=torch.bool))
 for si in range(2):
  seed=96061000+si;raw=sample(model,local,tx,10,cmd,[seed]*b,True,root_controls=controls)
  with torch.no_grad():pos,poses,root=fk(raw,canonical=False,return_pose=True)
  for i,c in enumerate(commands):
   path=out/'human'/f'{task}_s{si}_c{i}.npz';np.savez_compressed(path,motion=raw[i].numpy(),joints_zup_m=pos[i].numpy(),poses_axisangle=poses[i].numpy(),root_translation=root[i].numpy(),human_height=fk.height,fps=20.)
   q=pos[i,:,0];span=119/20.;quantity=float(torch.linalg.vector_norm(q[1:,:2]-q[:-1,:2],dim=-1).sum()/span)
   rows.append(dict(task=task,source=task+f'_s{si}',seed=seed,command=c,command_index=i,path=str(path),prompt=str(prompt),numeric_command='existing walk path-speed command; extra arm instruction supplied as part text',human_path_speed_m_s=quantity,source_type='FrankenMotion_generated',split='development',frames=120,generator=str(weight),generator_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),human_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
   print(task,si,c,round(quantity,3),flush=True)
  (out/'manifest.json').write_text(json.dumps(rows,indent=2))
(out/'protocol.json').write_text(json.dumps(dict(scope='New composition, existing numeric adapter, no new fine-tuning or output pose editing. 0.35 m/s is outside original 0.5–1.1 m/s numeric training range and is explicitly exploratory.',shared_weights=str(weight),rows=len(rows)),indent=2))
