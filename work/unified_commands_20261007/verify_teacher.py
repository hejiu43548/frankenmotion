from uc_common import *
from legacy_targets import LegacyTargets
from unified_control import load_base
from sampling import prepare,generate
from deployment_runtime import load_one_checkpoint
import gc
import table_goal_adapter_20261005 as ga
import reach_adapter_v6_20261005 as ra
import reach_exit_adapter_v6_20261005 as ea
torch.set_num_threads(2);teacher=load_base(LegacyTargets());results=[]
cases=[(t,[(lo+hi)/2],None) for t,(lo,hi) in zip(TASKS,RANGES)]+[('walk_endpoint',[1.4],[[1.4,.3]]),('place_hold_retract',[.4],[.84]),('back_departure',[.35],None),('side_departure',[.55],None),('root_profile',[0],[[.4,.8,-.15,.2,1,1]]),('turn_endpoint',[2.3],None)]
for kind,cs,extras in cases:
 src,local,tx,cmd,controls,c=prepare(kind,cs,0,extras)
 if kind in TASKS or kind=='root_profile':old=load_one_checkpoint(B/'current_full.pt')
 elif kind in ['walk_endpoint','turn_endpoint']:
  old=ga.load(B/'goal.pt').cpu();old.denoiser.goal=torch.tensor(extras) if kind=='walk_endpoint' else None
 elif kind=='place_hold_retract':
  old=ra.load(B/'reach.pt').cpu();old.denoiser.command=torch.tensor(list(zip(cs,extras)))
 else:
  old=ea.load(B/'exit.pt',reach_weight=B/'reach.pt').cpu();old.denoiser.condition=torch.tensor([[0 if kind=='back_departure' else 1,cs[0]]])
 enabled=kind not in ['root_profile','back_departure','side_departure'];rc=controls if kind in ['walk','back_walk','turn','walk_endpoint','turn_endpoint','root_profile'] else None
 actual=sample(old,local,tx,src['task_id'],cmd,[107082001],enabled,root_controls=rc);pred,_,_=generate(teacher,kind,cs,0,[107082001],extras);error=float((actual-pred).abs().max());fk=FK();poseerror=float((fk(actual)-fk(pred)).abs().max());results.append(dict(kind=kind,max_raw_error=error,max_joint_coordinate_error_m=poseerror));print(results[-1],flush=True)
 assert poseerror<.0005,(kind,error,poseerror)
 del old;gc.collect();torch.cuda.empty_cache()
(D/'teacher_parity.json').write_text(json.dumps(results,indent=2))
