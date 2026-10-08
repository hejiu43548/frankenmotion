from uc_common import *
from sampling import prepare,generate
from single_runtime import load_one_checkpoint
from shared_infer import command_features,generate as clean_generate
from conditions import KINDS
import argparse
p=argparse.ArgumentParser();p.add_argument('--checkpoint',required=True);a=p.parse_args();torch.set_num_threads(3);m=load_one_checkpoint(a.checkpoint);errs=[]
for kind in KINDS:
 extra=None
 if kind in TASKS:lo,hi=RANGES[TASKS.index(kind)];c=(lo+hi)/2
 elif kind=='walk_endpoint':c=1.8;extra=[1.8,.3]
 elif kind=='place_hold_retract':c=.4;extra=.84
 elif kind=='back_departure':c=.35
 elif kind=='side_departure':c=.55
 elif kind=='turn_endpoint':c=2.2
 else:c=0;extra=[.4,.8,-.2,.2,1,1]
 src,local,tx,_,_,cf=prepare(kind,[c],0,None if extra is None else [extra]);cc=command_features(kind,c,src['frames'],extra,fkheight=FK().height);de=float((cc-cf).abs().max());assert de<1e-6,(kind,de)
 x,_,_=generate(m,kind,[c],0,[107088999],None if extra is None else [extra]);y=clean_generate(m,local,tx,cc,107088999);err=float((x-y).abs().max());assert err<1e-6,(kind,err);errs.append(dict(kind=kind,feature_error=de,sample_error=err))
Path(a.checkpoint).with_name('standalone_parity.json').write_text(json.dumps(errs,indent=2));print('STANDALONE_PASS',max(r['sample_error'] for r in errs))
