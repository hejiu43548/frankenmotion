"""Standalone command generation. Only unified_generator.pt contains generator weights.
The .pt prompt cache holds text embeddings and the local text-assignment mask.
No source motion is accepted or used. FK/SMPL and tracker are separate stages.
"""
from pathlib import Path
import sys,argparse,json
import numpy as np,torch
from command_inputs import make,encode_control,KINDS,HH
from single_runtime import load_one_checkpoint

def command_features(kind,value,frames,extra=None,fkheight=1.372592926,device='cpu'):
 cmd=torch.tensor([value],device=device,dtype=torch.float32);phase=torch.linspace(0,1,frames,device=device)[None];ext=None if extra is None else torch.tensor([extra],device=device,dtype=torch.float32)
 if kind=='root_profile':
  v=torch.stack([ext[:,0,None]+phase*(ext[:,1]-ext[:,0])[:,None],ext[:,2,None]+phase*(ext[:,3]-ext[:,2])[:,None]],-1);ext=encode_control(v,ext[:,None,4:6].expand(1,frames,2))
 c=make(kind,cmd,phase,ext,frames=frames,fkheight=fkheight)
 if kind=='walk_endpoint':
  tt=torch.arange(frames,device=device)*.05;pr=torch.clamp((tt-.6)/.6,0,1)*torch.clamp((5.3-tt)/.8,0,1);pr/=pr[:-1].sum()*.05;v=torch.zeros(1,frames,2,device=device);v[:,:,0]=ext[:,0,None]*pr;v[:,:,1]=ext[:,1,None]*(tt<1.2).float()/1.2;c[:,:,18:22]=encode_control(v,torch.ones(1,frames,dtype=torch.bool,device=device))
 return c

@torch.no_grad()
def generate(model,local,tx,c,seed=0,steps=50):
 local=local[None] if local.ndim==2 else local;b,n,_=local.shape;device=local.device
 dummy=torch.cat([torch.zeros(b,n,205,device=device),local],-1);local=model.motion_normalizer(dummy)[...,205:];y=dict(mask=torch.ones(b,n,device=device,dtype=torch.bool),tx=model.prepare_tx_emb(tx));noise=torch.randn(b,n,205,generator=torch.Generator(device=device).manual_seed(seed),device=device);timeline=np.linspace(model.timesteps-1,0,steps,dtype=int);model.denoiser.static_residuals=model.denoiser.controller(c)
 try:
  for i,step in enumerate(timeline):
   x=torch.cat([noise,local],-1);pred=model.denoiser(x,y,torch.full((b,),int(step),device=device,dtype=torch.long))
   if i==len(timeline)-1:return model.motion_normalizer.inverse(pred)[...,:205]
   alpha=model.alphas_cumprod[step];next_alpha=model.alphas_cumprod[timeline[i+1]];eps=(x-alpha.sqrt()*pred)/(1-alpha).sqrt();noise=(next_alpha.sqrt()*pred+(1-next_alpha).sqrt()*eps)[...,:205]
 finally:model.denoiser.static_residuals=None

def main():
 p=argparse.ArgumentParser();p.add_argument('--repo',required=True);p.add_argument('--checkpoint',required=True);p.add_argument('--prompt-cache',required=True);p.add_argument('--kind',choices=KINDS,required=True);p.add_argument('--command',type=float,required=True);p.add_argument('--extra',type=float,nargs='+');p.add_argument('--output',required=True);p.add_argument('--seed',type=int,default=0);p.add_argument('--device',default='cpu');a=p.parse_args();sys.path.insert(0,a.repo);torch.set_num_threads(3);m=load_one_checkpoint(a.checkpoint,a.device);z=torch.load(a.prompt_cache,map_location=a.device,weights_only=False);n=z['local'].shape[0];extra=a.extra[0] if a.kind=='place_hold_retract' and a.extra else a.extra;c=command_features(a.kind,a.command,n,extra,device=a.device);raw=generate(m,z['local'],z['tx'],c,a.seed);np.savez_compressed(a.output,motion=raw[0].cpu().numpy(),control_features=c[0].cpu().numpy(),fps=20.,kind=a.kind,command=a.command,seed=a.seed)
if __name__=='__main__':main()
