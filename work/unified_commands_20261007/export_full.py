from uc_common import *
from unified_control import SharedCommands,load_base
from single_runtime import load_one_checkpoint
from conditions import KINDS
from sampling import generate
import argparse,hashlib
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);p.add_argument('--name',required=True);a=p.parse_args();out=D/'exports'/a.name;out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(3)
ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);ctrl=SharedCommands(ck['width']);ctrl.load_state_dict(ck['controller']);m=load_base(ctrl);old=torch.load(B/'current_full.pt',map_location='cpu',weights_only=False);cfg=old['diffusion_config']
for key in ['motion_normalizer','text_normalizer']:cfg[key]['_target_']='single_runtime.EmbeddedNormalizer'
f=out/'unified_generator.pt';torch.save(dict(kind='frankenmotion_all_commands_shared_v1',width=ck['width'],state_dict=m.state_dict(),diffusion_config=cfg,fields=FIELDS,intents=INTENTS,source_adapter_sha256=hashlib.sha256(Path(a.checkpoint).read_bytes()).hexdigest()),f)
loaded=[];old_load=torch.load
def traced(path,*args,**kwargs):loaded.append(str(path));return old_load(path,*args,**kwargs)
torch.load=traced
try:other=load_one_checkpoint(f)
finally:torch.load=old_load
assert loaded==[str(f)],loaded
errors={}
for kind in KINDS:
 extra=None
 if kind in TASKS:lo,hi=RANGES[TASKS.index(kind)];c=(lo+hi)/2
 elif kind=='walk_endpoint':c=1.8;extra=[[1.8,.3]]
 elif kind=='place_hold_retract':c=.4;extra=[.84]
 elif kind=='back_departure':c=.35
 elif kind=='side_departure':c=.55
 elif kind=='turn_endpoint':c=2.2
 else:c=0;extra=[[.4,.8,-.2,.2,1,1]]
 x,_,_=generate(m,kind,[c],0,[107088999],extra);y,_,_=generate(other,kind,[c],0,[107088999],extra);err=float((x-y).abs().max());assert err<1e-6;errors[kind]=err
(out/'export_audit.json').write_text(json.dumps(dict(sha256=hashlib.sha256(f.read_bytes()).hexdigest(),weight_file_loads=loaded,parity_errors=errors,controller_parameters=sum(p.numel() for p in ctrl.parameters()),controller_module_classes=sorted(set(type(x).__name__ for x in other.denoiser.modules())),scope='One bare frozen FrankenMotion denoiser and one dense shared conditional network. No legacy Root/Task/Goal/Reach/Exit module, no expert router. Tracker remains a separately frozen physical policy.'),indent=2));print('EXPORT_PASS',max(errors.values()),flush=True)
