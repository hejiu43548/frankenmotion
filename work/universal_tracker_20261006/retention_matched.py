"""Optional single-policy replay regularizer inside shared PPO optimizer steps.

Replay contains training states only. Targets always come from the same frozen
stable actor, never a per-task teacher selection. The deployed actor is unchanged.
"""
from pathlib import Path
import hashlib,json,numpy as np,torch
from tensordict import TensorDict


def attach_retention(runner, directory, weight=.05, batch_size=256):
    root=Path('/home/pku/frankenmotion')
    actor=runner.alg.actor
    input_dim=actor.mlp[0].in_features
    dataset=root/'outputs_amass/universal_tracker_20261006/long_retention_v2/dataset.npz'
    frozen=root/'outputs_amass/universal_tracker_20261006/backup/stable_frozen/actor.pt'
    with np.load(dataset) as z:
        keep=z['group']<42
        observations=torch.from_numpy(z['observations'][keep,:input_dim].copy()).float()
        groups=np.unique(z['group'][keep]).tolist()
    assert observations.shape[1]==input_dim and input_dim in (361,495) and max(groups)<42
    actor=runner.alg.actor
    device=next(actor.parameters()).device
    teacher=torch.jit.load(str(frozen),map_location=device).eval()
    generator=torch.Generator().manual_seed(6106001)
    stats={'calls':0,'loss_sum':0.,'last_loss':0.}

    def before_step(optimizer,args,kwargs):
        indices=torch.randint(len(observations),(batch_size,),generator=generator)
        obs=observations[indices].to(device)
        with torch.no_grad():target=teacher(obs[:,:361])
        with torch.enable_grad():
            prediction=actor(TensorDict({'actor':obs},batch_size=[batch_size]))
            error=(prediction-target).square().mean()
            (weight*error).backward()
        torch.nn.utils.clip_grad_norm_(actor.parameters(),runner.alg.max_grad_norm)
        stats['calls']+=1;stats['last_loss']=float(error.detach());stats['loss_sum']+=stats['last_loss']

    handle=runner.alg.optimizer.register_step_pre_hook(before_step)
    original_update=runner.alg.update
    def update():
        before_calls,before_sum=stats['calls'],stats['loss_sum']
        result=original_update()
        result['stable_replay_mse']=(stats['loss_sum']-before_sum)/max(stats['calls']-before_calls,1)
        return result
    runner.alg.update=update
    protocol=dict(student_input_dim=input_dim,teacher_input_dim=361,weight=weight,batch_size=batch_size,training_groups=groups,observations=len(observations),dataset=str(dataset),dataset_sha256=hashlib.sha256(dataset.read_bytes()).hexdigest(),teacher=str(frozen),teacher_sha256=hashlib.sha256(frozen.read_bytes()).hexdigest(),scope='Single frozen stable policy retention, injected into every shared PPO step; not per-class distillation. No final or validation groups.')
    (Path(directory)/'retention_protocol.json').write_text(json.dumps(protocol,indent=2))
    return handle,stats
