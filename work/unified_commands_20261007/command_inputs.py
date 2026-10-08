import torch,math
from command_schema import TASKS,RANGES,features
HH=1.2701193988323212
def encode_control(values,valid):
 if valid.ndim==2:valid=valid[...,None].expand_as(values)
 return torch.cat((values/values.new_tensor([3.,math.pi])*valid,valid.to(values)),-1)
KINDS=TASKS+['walk_endpoint','place_hold_retract','back_departure','side_departure','root_profile','turn_endpoint']
def make(kind,cmd,phase,extra=None,frames=None,fkheight=1.372592926):
 b,n=phase.shape;frames=frames or (60 if kind in ['raise_hand','reach','strike','kick','jump','lean'] else 120);device=phase.device;root=None;goal=None;height=None;intent=0;tid=TASKS.index(kind) if kind in TASKS else 11
 if kind in ['walk','back_walk','turn','turn_endpoint']:
  tid=4 if kind=='turn_endpoint' else tid;intent=1 if kind=='turn_endpoint' else 0;v=torch.zeros(b,n,2,device=device)
  if tid==4:v[:,:,1]=-cmd[:,None]/((frames-1)/20)
  else:v[:,:,0]=cmd[:,None]*fkheight/HH
  root=encode_control(v,torch.ones(b,n,device=device,dtype=torch.bool))
 if kind=='walk_endpoint':
  tid=10;intent=1;goal=extra;dist,angle=goal.unbind(-1);cmd=dist/4.3*HH/fkheight;tt=phase*5.95;grid=torch.arange(120,device=device)*.05;profile=torch.clamp((grid-.6)/.6,0,1)*torch.clamp((5.3-grid)/.8,0,1);norm=profile[:-1].sum()*.05;v=torch.zeros(b,n,2,device=device);v[:,:,0]=dist[:,None]*torch.clamp((tt-.6)/.6,0,1)*torch.clamp((5.3-tt)/.8,0,1)/norm;v[:,:,1]=angle[:,None]*(tt<1.2).to(v)/1.2;root=encode_control(v,torch.ones(b,n,device=device,dtype=torch.bool))
 if kind=='place_hold_retract':tid=1;intent=2;height=extra
 if kind in ['back_departure','side_departure']:tid=6 if kind=='back_departure' else 5;intent=3
 if kind=='root_profile':root=extra
 return features(tid,cmd,phase,intent,root,goal,height,frames)
