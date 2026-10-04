"""Calibrate appended preview statistics while preserving the loaded policy function."""
import torch
from unified_preview_20261004 import PREVIEW_DIM

@torch.no_grad()
def calibrate_preview(env,runner,resets=128,preview_dim=PREVIEW_DIM):
    sums={};squares={};count=0;retained=None
    for _ in range(resets):
        obs,_=env.reset();retained=obs
        for side in ['actor','critic']:
            x=obs[side][:,-preview_dim:].to('cpu',torch.float64)
            sums[side]=sums.get(side,torch.zeros(preview_dim,dtype=torch.float64))+x.sum(0)
            squares[side]=squares.get(side,torch.zeros(preview_dim,dtype=torch.float64))+(x*x).sum(0)
        count+=obs['actor'].shape[0]
    reports={}
    for side in ['actor','critic']:
        model=getattr(runner.alg,side);normalizer=model.obs_normalizer;first=model.mlp[0]
        random_input=normalizer._mean+torch.randn((1024,normalizer._mean.shape[-1]),device=normalizer._mean.device)*normalizer._std;random_before=model.mlp(normalizer(random_input)).clone()
        before=model(retained).clone();old_mean=normalizer._mean[:,-preview_dim:].clone();old_std=normalizer._std[:,-preview_dim:].clone()
        mean=(sums[side]/count).to(normalizer._mean).unsqueeze(0)
        variance=(squares[side]/count-(sums[side]/count).square()).clamp_min(.01**2).to(normalizer._var).unsqueeze(0);std=variance.sqrt()
        weight=first.weight[:,-preview_dim:].clone();eps=normalizer.eps
        # Affine compensation: W_old (x-m_old)/s_old == W_new (x-m_new)/s_new + bias_delta.
        first.bias.add_((weight*((mean-old_mean)/(old_std+eps))).sum(1))
        first.weight[:,-preview_dim:].copy_(weight*(std+eps)/(old_std+eps))
        normalizer._mean[:,-preview_dim:].copy_(mean);normalizer._var[:,-preview_dim:].copy_(variance);normalizer._std[:,-preview_dim:].copy_(std)
        after=model(retained);error=float((before-after).abs().max());assert torch.isfinite(after).all() and error<1e-4
        random_error=float((random_before-model.mlp(normalizer(random_input))).abs().max());assert random_error<1e-4
        reports[side]=dict(max_output_difference=error,random_input_count=1024,random_max_output_difference=random_error,new_mean_range=[float(mean.min()),float(mean.max())],new_std_range=[float(std.min()),float(std.max())],normalizer_count=int(normalizer.count))
    return dict(observations=count,reset_batches=resets,source='Training-corpus randomized reference-state initialization only; no validation samples',method='Only appended preview means/variances recalibrated. First layer affinely compensated to preserve loaded network outputs; existing feature statistics unchanged.',networks=reports)
