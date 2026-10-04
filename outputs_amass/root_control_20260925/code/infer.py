"""Numerical-control inference on a held-out annotation's text, without GT motion input."""
import argparse
from train import ROOT, load_model, move, sample, atomic_json
from control import labels
from pathlib import Path
from hydra.utils import instantiate
from omegaconf import OmegaConf
import torch
import numpy as np
p=argparse.ArgumentParser()
p.add_argument('--run',required=True);p.add_argument('--index',type=int,default=0)
p.add_argument('--speed',type=float,required=True);p.add_argument('--turn-deg',type=float,default=0)
p.add_argument('--seed',type=int,default=7301);p.add_argument('--output',required=True)
a=p.parse_args();run=Path(a.run)
cfg=OmegaConf.load(run/'base_config.yaml')
import json
provenance=json.loads((run/'provenance.json').read_text())
model,_=load_model(cfg,provenance['source_checkpoint'])
model.denoiser.load_adapter(torch.load(run/'best.pt',map_location='cuda',weights_only=False)['adapter']);model.eval()
ds=instantiate(cfg.data,split='val');ds.is_training=False
batch=move(ds.collate_fn([ds[a.index]]));target,valid=labels(batch['x'],batch['length'])
duration=(int(batch['length'][0])-1)/20
target[...,0]=a.speed;target[...,1]=np.deg2rad(a.turn_deg)/duration
raw=sample(model,batch,target,valid,random_seed=a.seed)
path=Path(a.output);path.parent.mkdir(parents=True,exist_ok=True)
np.save(path,raw[0].cpu().numpy())
atomic_json(path.with_suffix('.json'),{'keyid':batch['keyid'][0],'target_speed_mps':a.speed,
 'target_body_turn_deg':a.turn_deg,'duration_seconds':duration,'seed':a.seed,
 'note':'Requested controls; realized accuracy must be measured. Text is the selected validation annotation.'})
