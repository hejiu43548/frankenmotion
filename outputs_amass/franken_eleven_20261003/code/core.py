import os,sys,json
from pathlib import Path
ROOT=Path('/home/pku/frankenmotion')
OUT=Path(os.environ.get('ELEVEN_OUT',ROOT/'outputs_amass/franken_eleven_20261003'))
OLD=ROOT/'outputs_amass/root_control_20260925'
sys.path[:0]=[str(ROOT),str(OLD/'code')];os.chdir(ROOT)
import numpy as np
import torch
from torch import nn
from src.tools.geometry import rotation_6d_to_matrix,matrix_to_euler_angles,axis_angle_rotation,matrix_to_axis_angle
from control import RootControl,encode_control
HH=1.2701193988323212
TASKS=['raise_hand','reach','strike','wave','turn','sidestep','back_walk','kick','jump','lean','walk']
RANGES=[(.35,.75),(.25,.55),(1.5,2.5),(.08,.22),(.45,1.5),(.4,1.2),(.35,.9),(.25,.7),(.25,.55),(.4,.9),(.5,1.1)]
TOLS=[.02,.02,.1,.015,.08,.06,.05,.05,.04,.06,.05]
FRAMES=[60,60,60,120,120,120,120,60,60,60,120]
UNITS=['m','m','m/s','m','rad','m','m/s','m','m','rad','m/s']
def save(name,value):
    p=OUT/name;p.parent.mkdir(parents=True,exist_ok=True);t=p.with_suffix(p.suffix+'.tmp');t.write_text(json.dumps(value,indent=2,ensure_ascii=False));t.replace(p)

class FK:
    def __init__(self,device='cpu'):
        z=np.load(OUT/'skeleton.npz');self.J=torch.tensor(z['J'],device=device,dtype=torch.float32);self.parents=z['parents'][:22].tolist();self.height=float(z['height'])
    def __call__(self,raw,canonical=True,return_pose=False):
        # Same SMPL-RIFKE root reconstruction as upstream; no redundant xyz prediction.
        b,t,_=raw.shape;mat=rotation_6d_to_matrix(raw[...,4:136].reshape(b,t,22,6))
        e=matrix_to_euler_angles(mat[:,:,0],'ZYX')
        yaw=torch.cat([torch.zeros_like(raw[:,:1,3]),torch.cumsum(raw[:,:-1,3],1)],1)
        rz=axis_angle_rotation('Z',yaw);r0=rz@axis_angle_rotation('Y',e[...,1])@axis_angle_rotation('X',e[...,2])
        mats=torch.cat([r0[:,:,None],mat[:,:,1:]],2)
        v=(rz[...,:2,:2]@raw[...,1:3,None]).squeeze(-1)
        xy=torch.cat([torch.zeros_like(v[:,:1]),torch.cumsum(v[:,:-1],1)],1);trans=torch.cat([xy,raw[...,:1]],-1)
        g=[];p=[]
        for j in range(22):
            if j==0:g.append(mats[:,:,0]);p.append(self.J[0].expand(b,t,3)+trans)
            else:
                par=self.parents[j];g.append(g[par]@mats[:,:,j]);p.append(p[par]+(g[par]@(self.J[j]-self.J[par])[:,None]).squeeze(-1))
        p.extend([p[20]+(g[20]@(self.J[22]-self.J[20])[:,None]).squeeze(-1),p[21]+(g[21]@(self.J[37]-self.J[21])[:,None]).squeeze(-1)])
        p=torch.stack(p,2)
        if canonical:
            side=p[:,0,1,:2]-p[:,0,2,:2];angle=torch.atan2(side[:,1],side[:,0])-np.pi/2
            rot=axis_angle_rotation('Z',-angle);p=(rot[:,None,None]@p[...,None]).squeeze(-1)
        if return_pose:return p,matrix_to_axis_angle(mats).reshape(b,t,66),trans
        return p

def quantity(p,task,scale=1.,net_walk=False):
    # Frozen screenshot quantities, including legacy path-speed walk; net-speed is separately reported.
    task=TASKS[task] if isinstance(task,int) else task
    root=p[:,:,0];span=(p.shape[1]-1)/20
    if task=='raise_hand':q=torch.quantile((p[:,:,21]-root)[...,2],.95,dim=1)
    elif task=='reach':q=torch.quantile((p[:,:,21]-root)[...,0],.95,dim=1)
    elif task=='strike':
        w=p[:,:,21];s=.25*w[:,:-2]+.5*w[:,1:-1]+.25*w[:,2:];v=(s[:,2:]-s[:,:-2]).norm(dim=-1)*10;q=v[:,14:32].amax(1)
    elif task=='wave':
        a=p[:,16:101];lat=a[:,:,16]-a[:,:,17];lat=lat/lat.norm(dim=-1,keepdim=True).clamp_min(1e-8)
        s=((a[:,:,21]-(a[:,:,16]+a[:,:,17])/2)*lat).sum(-1);q=(torch.quantile(s,.95,dim=1)-torch.quantile(s,.05,dim=1))/2
    elif task=='turn':
        side=p[:,:,1,:2]-p[:,:,2,:2];yaw=torch.atan2(side[...,1],side[...,0]);diff=yaw[:,-1]-yaw[:,0];return -torch.atan2(torch.sin(diff),torch.cos(diff))
    elif task=='sidestep':q=-(root[:,:,1]-root[:,:1,1]).amin(1)
    elif task=='back_walk':q=-(root[:,-1,0]-root[:,0,0])/span
    elif task=='kick':s=(p[:,:,8]-root)[...,0];q=s[:,10:51].amax(1)-s[:,0]
    elif task=='jump':q=root[:,:,2].amax(1)-root[:,0,2]
    elif task=='lean':
        v=(p[:,:,16]+p[:,:,17])/2-root;pitch=torch.atan2(v[...,0],v[...,2]);return torch.quantile(pitch[:,20:59],.9,dim=1)
    elif task=='walk':q=(root[:,-1,0]-root[:,0,0])/span if net_walk else (root[:,1:,:2]-root[:,:-1,:2]).norm(dim=-1).sum(1)/span
    else:raise ValueError(task)
    return q*scale

class TaskControl(nn.Module):
    def __init__(self,root):
        super().__init__();self.root=root
        for p in root.parameters():p.requires_grad_(False)
        dim=root.base.latent_dim
        self.task=nn.Embedding(11,32)
        self.encoder=nn.Sequential(nn.Linear(33,128),nn.SiLU(),nn.Linear(128,dim))
        self.residuals=nn.ModuleList([nn.Sequential(nn.Linear(dim,128),nn.SiLU(),nn.Linear(128,dim)) for _ in root.base.seqTransEncoder.layers])
        for m in self.residuals:nn.init.zeros_(m[-1].weight);nn.init.zeros_(m[-1].bias)
        self.condition=None
        self.handles=[layer.register_forward_hook(self.hook(i)) for i,layer in enumerate(root.base.seqTransEncoder.layers)]
    def hook(self,i):
        def f(module,args,output):
            if self.condition is None:return output
            features,frames=self.condition;res=self.residuals[i](features)[:,None].expand(-1,frames,-1)
            return output+torch.nn.functional.pad(res,(0,0,output.shape[1]-frames,0))
        return f
    def forward(self,x,y,t,tf=None):
        self.condition=None
        if 'task_control' in y:
            ids,commands=y['task_control'];r=x.new_tensor(RANGES)[ids];value=((commands-r[:,0])/(r[:,1]-r[:,0])*2-1).clamp(-5,5)
            self.condition=(self.encoder(torch.cat([self.task(ids),value[:,None]],-1)),x.shape[1])
        try:return self.root(x,y,t,tf)
        finally:self.condition=None
    def train(self,mode=True):super().train(mode);self.root.eval();return self
    def adapter_state(self):return {k:v for k,v in self.state_dict().items() if not k.startswith('root.')}
    def load_adapter(self,s):
        result=self.load_state_dict(s,strict=False);assert not result.unexpected_keys;assert all(k.startswith('root.') for k in result.missing_keys)

def load_model(device='cuda',task_weights=None):
    import src.prepare
    from hydra.utils import instantiate
    from omegaconf import OmegaConf
    cfg=OmegaConf.load(OLD/'base_config.yaml');m=instantiate(cfg.diffusion)
    ck=torch.load(ROOT/'outputs_amass/official_20260916/logs/checkpoints/last.ckpt',map_location='cpu',weights_only=False);m.load_state_dict(ck['state_dict']);del ck
    m.denoiser=RootControl(m.denoiser);m.denoiser.load_adapter(torch.load(OLD/'best.pt',map_location='cpu',weights_only=False)['adapter'])
    m.denoiser=TaskControl(m.denoiser)
    if task_weights:m.denoiser.load_adapter(torch.load(task_weights,map_location='cpu',weights_only=False)['adapter'])
    return m.to(device).eval(),cfg

def make_y(m,x,tx,task,commands,use_task=True,root_controls=None):
    b,t,_=x.shape;y=dict(mask=torch.ones(b,t,device=x.device,dtype=torch.bool),tx=m.prepare_tx_emb(tx))
    if use_task:y['task_control']=(torch.full((b,),task,device=x.device,dtype=torch.long),commands)
    if root_controls is not None:y['root_control']=root_controls
    return y

@torch.no_grad()
def sample(m,local,tx,task,commands,seeds,use_task=True,steps=50,root_controls=None):
    b,t,_=local.shape;device=local.device
    # No ground-truth motion enters generation. Only cached text, noise and requested controls.
    dummy=torch.cat([torch.zeros(b,t,205,device=device),local],-1);norm=m.motion_normalizer(dummy);local=norm[...,205:]
    y=make_y(m,dummy,tx,task,commands,use_task,root_controls)
    noise=torch.stack([torch.randn(t,205,generator=torch.Generator(device=device).manual_seed(int(s)),device=device) for s in seeds])
    timeline=np.linspace(m.timesteps-1,0,steps,dtype=int)
    for i,step in enumerate(timeline):
        x=torch.cat([noise,local],-1);pred=m.denoiser(x,y,torch.full((b,),int(step),device=device,dtype=torch.long))
        if i==len(timeline)-1:return m.motion_normalizer.inverse(pred)[...,:205]
        a=m.alphas_cumprod[step];next_a=m.alphas_cumprod[timeline[i+1]]
        eps=(x-a.sqrt()*pred)/(1-a).sqrt();noise=(next_a.sqrt()*pred+(1-next_a).sqrt()*eps)[...,:205]
