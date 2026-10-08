from common import *
import argparse,hashlib
from omegaconf import OmegaConf
from deployment_runtime import load_one_checkpoint
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--name',required=True);a=p.parse_args();out=D/'exports'/a.name;out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2)
saved=torch.load(a.checkpoint,map_location='cpu',weights_only=False);width=saved['adapter']['residuals.0.0.weight'].shape[0]
if width==512:
 from wide_adapter import enable
 enable()
m,cfg=pa.core.load_model('cpu',task_weights=Path(a.checkpoint));state=m.state_dict();c=OmegaConf.to_container(cfg.diffusion,resolve=True)
for key in ['motion_normalizer','text_normalizer']:
 shape=list(state[key+'.mean'].shape);c[key]=dict(_target_='deployment_runtime.EmbeddedNormalizer',shape=shape,eps=c[key].get('eps',1e-12),disable=False)
f=out/'unified_generator.pt';torch.save(dict(kind='unified_full_frankenmotion_11',adapter_hidden_width=width,state_dict=state,diffusion_config=c,tasks=TASKS,ranges=RANGES,source_adapter_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest()),f)
loaded=[];old_load=torch.load
def traced(path,*args,**kwargs):loaded.append(str(path));return old_load(path,*args,**kwargs)
torch.load=traced
try:other=load_one_checkpoint(f)
finally:torch.load=old_load
assert loaded==[str(f)],loaded
fk=FK('cpu');errors=[]
for tid in range(11):
 src=next(r for r in SOURCES if r['task_id']==tid);lo,hi=RANGES[tid];local,tx,cmd,controls=inputs(src,[(lo+hi)/2],'cpu',fk);one=sample(m,local,tx,tid,cmd,[107079999],True,root_controls=controls);two=sample(other,local,tx,tid,cmd,[107079999],True,root_controls=controls);err=float((one-two).abs().max());assert err<1e-6,err;errors.append(err)
(out/'export_audit.json').write_text(json.dumps(dict(checkpoint=str(f),sha256=hashlib.sha256(f.read_bytes()).hexdigest(),weight_file_loads=loaded,parity_max_error=max(errors),all11_tasks_tested=True,selection='Export parity only, not final selection',base_and_root_and_adapter_and_normalizer_in_one_file=True),indent=2));print('export_parity',max(errors),flush=True)
