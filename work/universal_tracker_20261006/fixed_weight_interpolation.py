"""Development diagnostic: one fixed merged actor, never per-task routing.
Normalize first-layer coordinates before interpolating aligned network weights.
"""
import copy,json,hashlib,argparse
from pathlib import Path
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/universal_tracker_20261006';torch.set_num_threads(2);torch.manual_seed(610699)
parser=argparse.ArgumentParser();parser.add_argument('--name',default='fixed_weight_interpolation');parser.add_argument('--fractions',nargs='+',type=float,default=[.25,.5,.75]);args=parser.parse_args();assert all(0<=x<=1 for x in args.fractions)
stable_path=D/'backup/stable_frozen/policy.pt';candidate_path=D/'training/joint_v3_broad_init/model_3000.pt';stable=torch.load(stable_path,map_location='cpu',weights_only=False);candidate=torch.load(candidate_path,map_location='cpu',weights_only=False);a=stable['actor_state_dict'];b=candidate['actor_state_dict'];assert a['mlp.0.weight'].shape==b['mlp.0.weight'].shape
dim=a['mlp.0.weight'].shape[1];dummy=TensorDict({'actor':torch.zeros(1,dim)},batch_size=[1]);model=MLPModel(dummy,{'actor':['actor']},'actor',29,hidden_dims=[512,256,128],activation='elu',obs_normalization=True,stochastic=True);model.eval();eps=model.obs_normalizer.eps
obs=torch.cat([s['obs_normalizer._mean']+torch.randn(1024,dim)*s['obs_normalizer._std'] for s in [a,b]])
def forward(state):
 model.load_state_dict(state,strict=True)
 with torch.no_grad():return model(TensorDict({'actor':obs},batch_size=[len(obs)])).clone()
before=forward(b);aligned=copy.deepcopy(b);w=b['mlp.0.weight'];oldmean=b['obs_normalizer._mean'];oldstd=b['obs_normalizer._std'];newmean=a['obs_normalizer._mean'];newstd=a['obs_normalizer._std'];aligned['mlp.0.bias']=b['mlp.0.bias']+(w*((newmean-oldmean)/(oldstd+eps))).sum(1);aligned['mlp.0.weight']=w*(newstd+eps)/(oldstd+eps)
for k in a:
 if k.startswith('obs_normalizer.'):aligned[k]=a[k].clone()
error=float((forward(aligned)-before).abs().max());assert error<1e-4,error
out=D/args.name;out.mkdir(exist_ok=False);records=[]
for fraction in args.fractions:
 checkpoint=copy.deepcopy(candidate);checkpoint['actor_state_dict']={k:a[k].clone() if k.startswith('obs_normalizer.') else ((1-fraction)*a[k]+fraction*aligned[k]) for k in a};checkpoint.pop('optimizer_state_dict',None);checkpoint['iter']=0;checkpoint['infos']=dict(scope='Inference-only fixed weight interpolation; optimizer intentionally omitted; no per-task routing',candidate_fraction=fraction)
 path=out/f'candidate_{fraction:.2f}.pt';torch.save(checkpoint,path);records.append(dict(path=str(path),candidate_fraction=fraction,sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
(out/'protocol.json').write_text(json.dumps(dict(stable=str(stable_path),stable_sha256=hashlib.sha256(stable_path.read_bytes()).hexdigest(),candidate=str(candidate_path),candidate_sha256=hashlib.sha256(candidate_path.read_bytes()).hexdigest(),normalization_alignment_max_action_error=error,observations_tested=len(obs),scope='Additional development baseline, one constant fraction for all layers/tasks/phases. This is not joint training and will be identified separately. Not a per-task mixture or ensemble at runtime.',records=records),indent=2));print(error,records)
