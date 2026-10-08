"""Development demo candidates: original FrankenMotion text and optional root-speed command.
No mocap frames, output-pose editing, or task-specific generator checkpoint.
"""
import os,sys,json,hashlib,time
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/universal_tracker_20261006'
sys.path[:0]=[str(R/'outputs_amass/franken_eleven_20261003/code')]
from core import torch,np,load_model,FK,sample,encode_control,HH,OLD
from hydra.utils import instantiate
from omegaconf import OmegaConf
from src.tools.parse_user_input import parse_and_validate_user_input
from src.data.text_part_utils import load_from_annotation_with_model
from src.data.text_motion import align
out=D/'generated_demo_candidates';out.mkdir(exist_ok=False);(out/'prompts').mkdir();(out/'human').mkdir();torch.set_num_threads(4)
def part(text,start=0,end=6):return dict(text=text,start=start,end=end)
def base():return {k:[part(v)] for k,v in dict(head='look forward',spine='upright',left_arm='relaxed',right_arm='relaxed',left_leg='stand',right_leg='stand',trajectory='stand still').items()}
cases=[]
for task,caption,arm in [('walk_wave','A person walks forward while waving hello with the right hand.','wave hello repeatedly'),('walk_point','A person walks forward while pointing ahead with the right hand.','point forward')]:
 b=base();b.update(action=[part('walk forward and '+arm)],trajectory=[part('walk forward')],left_leg=[part('walk forward')],right_leg=[part('walk forward')],left_arm=[part('swing naturally')],right_arm=[part('relaxed',0,.5),part(arm,.5,5.3),part('lower hand',5.3,6)])
 cases.append(dict(task=task,caption=caption,body_parts=b,commands=[.35,.55],numeric_command='root forward speed in human metres per second'))
for task,caption,changes in [
 ('squat','A person performs a controlled squat then stands upright.',dict(action=[part('squat and stand')],left_leg=[part('stand',0,.6),part('bend knees into a squat',.6,2.2),part('hold squat',2.2,3.2),part('stand up',3.2,5),part('stand',5,6)],right_leg=[part('stand',0,.6),part('bend knees into a squat',.6,2.2),part('hold squat',2.2,3.2),part('stand up',3.2,5),part('stand',5,6)],left_arm=[part('reach forward to balance')],right_arm=[part('reach forward to balance')])),
 ('stretch','A person raises both arms overhead to stretch and lowers them.',dict(action=[part('stretch both arms overhead')],left_arm=[part('raise arm overhead',0,2),part('stretch overhead',2,4),part('lower arm',4,6)],right_arm=[part('raise arm overhead',0,2),part('stretch overhead',2,4),part('lower arm',4,6)])),
 ('clap','A standing person claps both hands several times in front of the chest.',dict(action=[part('clap hands')],left_arm=[part('clap hands repeatedly')],right_arm=[part('clap hands repeatedly')])),
 ('balance','A person lifts the right knee, balances on the left leg, and lowers the right foot.',dict(action=[part('balance on left leg')],right_leg=[part('stand',0,.6),part('raise knee',.6,1.8),part('hold knee raised',1.8,4),part('lower foot',4,5.4),part('stand',5.4,6)],left_arm=[part('extend arm sideways for balance')],right_arm=[part('extend arm sideways for balance')])),
 ('lunge','A person steps forward into a gentle right lunge then returns to standing.',dict(action=[part('right forward lunge and return')],left_leg=[part('stand',0,.5),part('extend leg behind',.5,2.5),part('return to standing',2.5,6)],right_leg=[part('stand',0,.5),part('step forward and bend knee',.5,2.5),part('step back and stand',2.5,6)])),
 ('bow','A person gives a gentle forward bow, straightens up, and waves hello.',dict(action=[part('bow',0,3),part('wave hello',3,6)],spine=[part('lean forward',0,1.3),part('hold a bow',1.3,2),part('straighten up',2,3),part('upright',3,6)],right_arm=[part('relaxed',0,3),part('wave hello',3,6)]))]:
 b=base();b.update(changes);cases.append(dict(task=task,caption=caption,body_parts=b,commands=[None],numeric_command=None))
cfg=OmegaConf.load(OLD/'base_config.yaml');enc=instantiate(cfg.data.text_encoder);enc.no_model=False;enc.rand_mask=False
for case in cases:
 path=out/'prompts'/(case['task']+'.json');path.write_text(json.dumps(dict(duration=6.,sequence_caption=case['caption'],body_parts=case['body_parts']),indent=2))
 ann=parse_and_validate_user_input(str(path),cfg=cfg,fps=20);emb,_=load_from_annotation_with_model(enc,ann['annotations'],ann['path'],ann['start'],ann['end']);local=align(torch.zeros(120,205),emb['local']['x']);tx={'x':emb['x'][None],'length':torch.tensor([emb['length']])};torch.save(dict(local=local,tx=tx),path.with_suffix('.pt'));case['prompt']=str(path)
print('Text encoding complete',len(cases),flush=True);del enc
model,_=load_model(device='cpu');fk=FK('cpu');manifest=[];start=time.monotonic()
for case in cases:
 cache=torch.load(Path(case['prompt']).with_suffix('.pt'),map_location='cpu',weights_only=False)
 for seed_index in range(2):
  seed=96061000+seed_index
  for ci,command in enumerate(case['commands']):
   controls=None
   if command is not None:
    values=torch.zeros(1,120,2);values[...,0]=command;controls=encode_control(values,torch.ones(1,120,dtype=torch.bool))
   raw=sample(model,cache['local'][None],cache['tx'],0,torch.zeros(1),[seed],use_task=False,root_controls=controls)
   with torch.no_grad():pos,poses,root=fk(raw,canonical=False,return_pose=True)
   path=out/'human'/f'{case["task"]}_s{seed_index}_c{ci}.npz';np.savez_compressed(path,motion=raw[0].numpy(),joints_zup_m=pos[0].numpy(),poses_axisangle=poses[0].numpy(),root_translation=root[0].numpy(),human_height=fk.height,fps=20.)
   manifest.append(dict(task=case['task'],source=case['task']+f'_s{seed_index}',seed=seed,command=command,command_index=ci,path=str(path),prompt=case['prompt'],caption=case['caption'],numeric_command=case['numeric_command'],source_type='FrankenMotion_generated_from_text_and_noise',split='development',frames=120,human_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
   (out/'manifest.json').write_text(json.dumps(manifest,indent=2));print(len(manifest),case['task'],command,'elapsed',round(time.monotonic()-start,1),flush=True)
(out/'protocol.json').write_text(json.dumps(dict(cases=cases,weights={str(p):hashlib.sha256(p.read_bytes()).hexdigest() for p in [R/'outputs_amass/official_20260916/logs/checkpoints/last.ckpt',OLD/'best.pt']},sampler='50-step DDIM; CPU',scope='Development candidates, not blind test. Same generator weights for every category. Text-only classes do not claim new numeric control. No reference pose editing.'),indent=2))
