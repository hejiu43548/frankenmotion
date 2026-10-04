"""Preserve the original policy exactly while adding initially ignored preview inputs."""
import argparse,torch,hashlib,json
from pathlib import Path
from unified_preview_20261004 import OFFSETS,LONG_OFFSETS
p=argparse.ArgumentParser();p.add_argument('--source',required=True);p.add_argument('--destination',required=True);p.add_argument('--long-preview',action='store_true');a=p.parse_args();offsets=LONG_OFFSETS if a.long_preview else OFFSETS;preview_dim=67*len(offsets);src=Path(a.source);dst=Path(a.destination);assert not dst.exists();x=torch.load(src,map_location='cpu',weights_only=False);legacy='model_state_dict' in x;checks=[]
for side in ['actor','critic']:
 d=x['model_state_dict'] if legacy else x[side+'_state_dict'];key=side+'.0.weight' if legacy else 'mlp.0.weight';prefix=side+'_obs_normalizer.' if legacy else 'obs_normalizer.';w=d[key].clone();extra_dim=(160 if side=='actor' else 286)+preview_dim-w.shape[1];assert extra_dim>=0;d[key]=torch.cat([w,torch.zeros((w.shape[0],extra_dim),dtype=w.dtype)],1)
 for suffix,value in [('_mean',0.),('_var',1.),('_std',1.)]:
  k=prefix+suffix;t=d[k];d[k]=torch.cat([t,torch.full((1,extra_dim),value,dtype=t.dtype)],1)
 inp=torch.randn(32,w.shape[1]);extra=torch.randn(32,extra_dim);err=(inp@w.T-torch.cat([inp,extra],1)@d[key].T).abs().max().item();assert err<1e-5;checks.append(dict(network=side,old_inputs=w.shape[1],new_inputs=d[key].shape[1],initial_first_layer_max_error=err))
x['iter']=0;x['optimizer_state_dict']={};dst.parent.mkdir(parents=True,exist_ok=True);torch.save(x,dst);dst.with_suffix('.json').write_text(json.dumps(dict(source=str(src),source_sha256=hashlib.sha256(src.read_bytes()).hexdigest(),output_sha256=hashlib.sha256(dst.read_bytes()).hexdigest(),preview_dim=preview_dim,preview_offsets=list(offsets),checks=checks),indent=2));print(json.dumps(checks))
