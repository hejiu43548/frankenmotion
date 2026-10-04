"""Export one deterministic actor with its normalization; no critic or task router."""
import argparse,hashlib,json
from pathlib import Path
import torch
from tensordict import TensorDict
from rsl_rl.models import MLPModel
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--output',required=True);a=p.parse_args()
torch.set_num_threads(2);torch.manual_seed(6104)
source=Path(a.checkpoint);output=Path(a.output);assert not output.exists()
checkpoint=torch.load(source,map_location='cpu',weights_only=False)
state=checkpoint['actor_state_dict'];dim=state['mlp.0.weight'].shape[1];assert dim in [160,361,495]
dummy=TensorDict({'actor':torch.zeros(1,dim)},batch_size=[1])
actor=MLPModel(dummy,{'actor':['actor']},'actor',29,hidden_dims=[512,256,128],activation='elu',obs_normalization=True,stochastic=True)
actor.load_state_dict(state,strict=True);actor.eval();exported=torch.jit.script(actor.as_jit().eval())
output.parent.mkdir(parents=True,exist_ok=True);exported.save(str(output));loaded=torch.jit.load(str(output)).eval()
mean=state['obs_normalizer._mean'];std=state['obs_normalizer._std'];observations=mean+torch.randn(512,dim)*std
with torch.inference_mode():
    expected=actor(TensorDict({'actor':observations},batch_size=[512]));actual=loaded(observations)
error=float((expected-actual).abs().max());assert error<1e-6 and torch.isfinite(actual).all()
report=dict(checkpoint=str(source),checkpoint_sha256=hashlib.sha256(source.read_bytes()).hexdigest(),actor_sha256=hashlib.sha256(output.read_bytes()).hexdigest(),preview_offsets={160:[],361:[5,10,20],495:[5,10,20,35,50]}[dim],input_dimensions=dim,output_dimensions=29,random_observations=512,max_absolute_action_error=error,contains='one actor and fixed observation normalization; no critic, task ID, or routing',contract='Expects concatenated BM actor observations in the trained order, with optional shared reference preview. Outputs normalized actions; the existing native action scale/offset, limits, and observation construction remain required. This export is not a hardware deployment validation.')
output.with_suffix('.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
