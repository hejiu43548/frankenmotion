"""Text-only bow candidates without the T-pose in the earlier official caption.
Presentation prompt refinement after policy freeze; never enters benchmark scores. Action phrase changed from bow to bend forward at the waist; no claim of an isolated causal language experiment.
"""
import sys,json,hashlib
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/universal_tracker_20261006';sys.path.insert(0,str(R/'outputs_amass/franken_eleven_20261003/code'))
from core import torch,np,FK,load_model,sample,OLD
from hydra.utils import instantiate
from omegaconf import OmegaConf
from src.data.text_part_utils import load_from_annotation_with_model
from src.data.text_motion import align
out=D/'generated_clean_bend';out.mkdir(exist_ok=False);(out/'prompts').mkdir();(out/'human').mkdir();torch.set_num_threads(4);duration=5.;annotations=[]
def add(part,text,start,end):annotations.append(dict(bodypart=part,text=text,start=start,end=end,confidence=5,reasoning='User-facing presentation command assembled from standard part-language; no source motion read.'))
caption='A person stands with both arms relaxed at their sides, bows forward politely, and returns to an upright standing position.'
add('sequence_caption',caption,0,duration)
for part in ['left_arm','right_arm']:add(part,'hanging relaxed at the side',0,duration)
for part in ['left_leg','right_leg']:add(part,'standing',0,duration)
add('trajectory','stationary',0,duration)
for start,end,head,spine,action in [(0,1,'facing forward','upright','stand'),(1,3,'looking down','bending forward','bend forward at the waist'),(3,5,'facing forward','upright','stand')]:
 add('head',head,start,end);add('spine',spine,start,end);add('action',action,start,end)
row=dict(path='authored/stand_bow_stand',start=0.,end=duration,duration=duration,caption_label=caption,annotations=annotations);prompt=out/'prompts/clean_bow.json';prompt.write_text(json.dumps(row,indent=2));cfg=OmegaConf.load(OLD/'base_config.yaml');enc=instantiate(cfg.data.text_encoder);enc.no_model=False;enc.rand_mask=False;emb,_=load_from_annotation_with_model(enc,annotations,row['path'],0,duration);local=align(torch.zeros(100,205),emb['local']['x']);tx={'x':emb['x'][None],'length':torch.tensor([emb['length']])};torch.save(dict(local=local,tx=tx),prompt.with_suffix('.pt'));del enc
model,_=load_model('cpu');fk=FK('cpu');rows=[]
for si in range(2):
 seed=107069000+si;raw=sample(model,local[None],tx,0,torch.zeros(1),[seed],False)
 with torch.no_grad():pos,poses,root=fk(raw,canonical=False,return_pose=True)
 path=out/'human'/f'clean_bend_s{si}.npz';np.savez_compressed(path,motion=raw[0].numpy(),joints_zup_m=pos[0].numpy(),poses_axisangle=poses[0].numpy(),root_translation=root[0].numpy(),human_height=fk.height,fps=20.);rows.append(dict(task='bow',source=f'clean_bend_s{si}',seed=seed,command=None,path=str(path),frames=100,prompt=str(prompt),caption=caption,source_type='FrankenMotion_generated_from_authored_timed_text_only',split='presentation_after_freeze',human_sha256=hashlib.sha256(path.read_bytes()).hexdigest()));(out/'manifest.json').write_text(json.dumps(rows,indent=2));print(rows[-1]['source'],flush=True)
(out/'protocol.json').write_text(json.dumps(dict(scope=__doc__,candidates=2,source_motion_read=False,pose_editing=False,numeric_bow_angle_control=False,sampler='50 DDIM steps,guidance1,CPU',model=str(R/'outputs_amass/official_20260916/logs/checkpoints/last.ckpt'),frozen_tracker_sha256=json.loads((D/'frozen_unified/protocol.json').read_text())['checkpoint_sha256']),indent=2))
