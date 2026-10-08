import sys,json
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/reach_demo_20261005';sys.path[:0]=[str(R/'work'),str(R/'outputs_amass/franken_eleven_20261003/code')]
import core
from core import torch
from hydra.utils import instantiate
from omegaconf import OmegaConf
from src.tools.parse_user_input import parse_and_validate_user_input
from src.data.text_part_utils import load_from_annotation_with_model
from src.data.text_motion import align
import src.prepare
out=D/'prompts';out.mkdir(exist_ok=True);torch.set_num_threads(2);cfg=OmegaConf.load(core.OLD/'base_config.yaml');enc=instantiate(cfg.data.text_encoder);enc.no_model=False;enc.rand_mask=False
for i,cap in enumerate(['A person reaches forward with the right hand, places the hand on a table, holds, then retracts and lowers the arm.','A standing person places their right hand forward on a tabletop, pauses, and returns the arm to their side.']):
 def a(text,start=0,end=6.):return dict(text=text,start=start,end=end)
 parts={k:[a(v)] for k,v in [('head','look forward'),('spine','stand upright'),('left_leg','stand still'),('right_leg','stand still'),('left_arm','relaxed at side'),('trajectory','stand still')]};parts['action']=[a('reach forward and place hand',0,2.4),a('hold hand on table',2.4,3.6),a('retract and lower arm',3.6,6)];parts['right_arm']=[a('raise hand',0,.8),a('reach forward',.8,1.8),a('lower hand onto tabletop',1.8,2.4),a('hold hand on table',2.4,3.6),a('lift hand off table',3.6,4.2),a('retract hand',4.2,5.2),a('lower arm to side',5.2,6)];p=out/f'reach_place_p{i}.json';p.write_text(json.dumps(dict(duration=6.,sequence_caption=cap,body_parts=parts),indent=2));ann=parse_and_validate_user_input(str(p),cfg=cfg,fps=20);emb,_=load_from_annotation_with_model(enc,ann['annotations'],ann['path'],ann['start'],ann['end']);local=align(torch.zeros(120,205),emb['local']['x']);tx={'x':emb['x'][None],'length':torch.tensor([emb['length']])};torch.save(dict(local=local,tx=tx),p.with_suffix('.pt'));print(p,flush=True)
