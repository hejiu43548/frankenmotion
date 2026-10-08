from uc_common import *
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

def random_batch(per_kind=16,device='cuda',generator=None,boundaries=False):
 rand=lambda *shape:torch.rand(*shape,device=device,generator=generator);cs=[];groups=[]
 for k,kind in enumerate(KINDS):
  phase=rand(per_kind,1);extra=None
  if boundaries:
   mask=rand(per_kind,1)<.25;phase=torch.where(mask,(rand(per_kind,1)>.5).float(),phase)
  if k<11:
   lo,hi=RANGES[k];cmd=lo+(hi-lo)*rand(per_kind)
  elif kind=='walk_endpoint':extra=torch.stack([.8+2*rand(per_kind),-.65+1.3*rand(per_kind)],-1);cmd=extra[:,0]
  elif kind=='place_hold_retract':cmd=.28+.24*rand(per_kind);extra=.81+.06*rand(per_kind)
  elif kind=='back_departure':cmd=.28+.14*rand(per_kind)
  elif kind=='side_departure':cmd=.4+.3*rand(per_kind)
  elif kind=='turn_endpoint':cmd=1.5+3*rand(per_kind)
  else:
   cmd=rand(per_kind)*0;values=torch.stack([3*rand(per_kind,1),(rand(per_kind,1)*2-1)*math.pi],-1);valid=rand(per_kind,1,2)>.15;extra=encode_control(values,valid)
  c=make(kind,cmd,phase,extra)
  if boundaries and kind=='root_profile':c[:,:,45]=.25+3.75*rand(per_kind,1)
  cs.append(c);groups.append(torch.full((per_kind,),k,device=device,dtype=torch.long))
 return torch.cat(cs,0).squeeze(1),torch.cat(groups)
