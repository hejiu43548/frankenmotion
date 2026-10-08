"""Verify every final candidate export carries the exact same tensors and outputs."""
import json,hashlib
from pathlib import Path
import torch
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/universal_tracker_20261006';f=json.loads((D/'frozen_unified/protocol.json').read_text());torch.set_num_threads(2);torch.manual_seed(610609)
source=D/'frozen_unified/actor.pt';assert hashlib.sha256(source.read_bytes()).hexdigest()==f['actor_sha256'];reference=torch.jit.load(str(source),map_location='cpu').eval();state=reference.state_dict();checkpoint=torch.load(f['checkpoint'],map_location='cpu',weights_only=False)['actor_state_dict'];mean=checkpoint['obs_normalizer._mean'];std=checkpoint['obs_normalizer._std'];obs=mean+torch.randn(512,mean.shape[-1])*std
gpu=json.loads((D/'evaluation/fresh_gpu_candidate/protocol.json').read_text());assert gpu['sha256']==f['checkpoint_sha256'] and not gpu['task_routing']
paths=[D/'general_evaluation/fresh_candidate/actor.pt',D/'general_evaluation/fresh_candidate_natural/actor.pt',D/'table_evaluation/fresh_candidate/actor.pt']
paths.append(D/'general_evaluation/long_horizon_delay_20ms/actor.pt');paths+=list((D/'general_evaluation').glob('fresh_robust_candidate_*/actor.pt'));paths+=list((D/'general_evaluation').glob('frozen_demo_*/actor.pt'));results=[]
with torch.inference_mode():
 expected=reference(obs)
 for path in paths:
  assert path.exists(),path
  metadata=json.loads(path.with_suffix('.json').read_text());assert metadata['checkpoint_sha256']==f['checkpoint_sha256']
  actor=torch.jit.load(str(path),map_location='cpu').eval();other=actor.state_dict();assert state.keys()==other.keys();assert all(torch.equal(state[k],other[k]) for k in state),path;error=float((actor(obs)-expected).abs().max());assert error==0
  results.append(dict(path=str(path),serialization_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),exact_state_tensor_match=True,random_observation_max_absolute_error=error))
(D/'single_policy_verification.json').write_text(json.dumps(dict(checkpoint_sha256=f['checkpoint_sha256'],frozen_actor_sha256=f['actor_sha256'],actors=len(results),random_observations=512,all_identical_tensors=True,results=results,scope='TorchScript archive byte hashes can differ between exports; all trainable/normalization tensors and evaluated outputs are exactly identical. One candidate policy across tasks/scenes/robustness profiles. Baseline controllers are separate comparison systems.'),indent=2));print('Verified',len(results),'identical candidate actors')
