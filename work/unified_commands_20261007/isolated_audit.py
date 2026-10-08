from pathlib import Path
import sys,json,argparse,torch
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--prompt',required=True);a=p.parse_args();sys.path.insert(0,'/home/pku/frankenmotion')
from single_runtime import load_one_checkpoint
from shared_infer import command_features,generate
torch.set_num_threads(3);files=[];old=torch.load
def traced(path,*args,**kw):files.append(str(path));return old(path,*args,**kw)
torch.load=traced;m=load_one_checkpoint(a.checkpoint);assert files==[a.checkpoint];z=torch.load(a.prompt,map_location='cpu',weights_only=False);c=command_features('walk',.8,z['local'].shape[0]);raw=generate(m,z['local'],z['tx'],c,107089991);assert torch.isfinite(raw).all();forbidden=['control','core','physical_adapter_20261003','deployment_runtime','legacy_targets','uc_common'];assert not any(n in sys.modules for n in forbidden),[n for n in forbidden if n in sys.modules]
result=dict(fresh_process=True,weight_files=[files[0]],prompt_cache_files=files[1:],legacy_modules_imported=[],shape=list(raw.shape),all_finite=True);Path(a.checkpoint).with_name('isolated_audit.json').write_text(json.dumps(result,indent=2));print(result)
