from uc_common import *
from conditions import make

def source(kind,pi=0):
 if kind=='place_hold_retract':return dict(task='reach',task_id=1,frames=120,prompt=str(R/f'outputs_amass/reach_demo_20261005/prompts/reach_place_p{pi%2}.json'),source=f'reach_place_p{pi%2}_s0')
 task={'walk_endpoint':'walk','root_profile':'walk','turn_endpoint':'turn','back_departure':'back_walk','side_departure':'sidestep'}.get(kind,kind)
 return next(r for r in prev.SOURCES if r['task']==task and f'_p{pi}_' in r['source'])
def prepare(kind,commands,pi=0,extras=None,device='cpu'):
 src=source(kind,pi);fk=FK(device);cmd=torch.as_tensor(commands,device=device,dtype=torch.float32);n=src['frames'];b=len(cmd);phase=torch.linspace(0,1,n,device=device)[None].expand(b,-1);extra=None if extras is None else torch.tensor(extras,device=device,dtype=torch.float32)
 if kind=='walk_endpoint':cmd=extra[:,0]/4.3*HH/fk.height
 local,tx,cmd,controls=prev.inputs(src,cmd.tolist(),device,fk)
 if kind=='walk_endpoint':
  tt=torch.arange(n,device=device)*.05;profile=torch.clamp((tt-.6)/.6,0,1)*torch.clamp((5.3-tt)/.8,0,1);profile/=profile[:-1].sum()*.05;values=torch.zeros(b,n,2,device=device);values[:,:,0]=extra[:,0,None]*profile;values[:,:,1]=extra[:,1,None]*(tt<1.2).float()/1.2;controls=encode_control(values,torch.ones(b,n,device=device,dtype=torch.bool))
 if kind=='root_profile':
  # extras = (speed_start,speed_end,yaw_rate_start,yaw_rate_end,valid_speed,valid_yaw)
  values=torch.stack([extra[:,0,None]+phase*(extra[:,1]-extra[:,0])[:,None],extra[:,2,None]+phase*(extra[:,3]-extra[:,2])[:,None]],-1);valid=extra[:,None,4:6].expand(b,n,2);controls=encode_control(values,valid);extra=controls
 c=make(kind,cmd,phase,extra,frames=n,fkheight=fk.height)
 if kind in ['walk','back_walk','turn','walk_endpoint','turn_endpoint','root_profile']:c[:,:,18:22]=controls
 return src,local,tx,cmd,controls,c

def generate(m,kind,commands,pi,seeds,extras=None,device='cpu'):
 src,local,tx,cmd,controls,c=prepare(kind,commands,pi,extras,device);m.denoiser.command_features=c

 with torch.no_grad():m.denoiser.static_residuals=m.denoiser.controller(c)
 try:return sample(m,local,tx,src['task_id'],cmd,seeds,False,steps=50),src,c
 finally:m.denoiser.static_residuals=None
