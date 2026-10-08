from uc_common import *
from unified_control import SharedCommands,load_base
from legacy_targets import LegacyTargets
from sampling import generate,source
from conditions import KINDS
import argparse,time,hashlib
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--checkpoint');p.add_argument('--full',action='store_true');p.add_argument('--teachers',action='store_true');p.add_argument('--split',default='development',choices=['development','final']);a=p.parse_args();assert a.teachers!=bool(a.checkpoint);out=D/'generation'/a.name;out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(3)
if a.teachers:controller=LegacyTargets();weight='frozen_training_baseline';weight_hash='see_backup_manifest'
elif a.full:
 from single_runtime import load_one_checkpoint
 m=load_one_checkpoint(a.checkpoint);weight=str(a.checkpoint);weight_hash=hashlib.sha256(Path(weight).read_bytes()).hexdigest()
else:
 ck=torch.load(a.checkpoint,map_location='cpu',weights_only=False);controller=SharedCommands(ck['width']);controller.load_state_dict(ck['controller']);weight=str(a.checkpoint);weight_hash=hashlib.sha256(Path(weight).read_bytes()).hexdigest()
if not a.full:m=load_base(controller)
fk=FK();rows=[];start=time.time();pis=[0,2] if a.split=='development' else [0,1,2,3];ns=1 if a.split=='development' else 4;seedbase=107083000 if a.split=='development' else 107084000
(out/'protocol.json').write_text(json.dumps(dict(split=a.split,checkpoint=weight,sha256=weight_hash,seed_base=seedbase,steps=50,kinds=KINDS,scope='Baseline preserves frozen legacy controls; student has only one shared control network. No command optimization or post-sampling motion edits. Generic flat-scene transfer and table contact evaluated separately.'),indent=2))
for kind in KINDS:
 for pi in pis:
  for si in range(ns):
   extras=None
   if kind in TASKS:
    lo,hi=RANGES[TASKS.index(kind)];cs=np.linspace(lo,hi,5).tolist()
   elif kind=='walk_endpoint':extras=[[d,ang] for d in [1.,1.8,2.6] for ang in [-.55,0,.55]];cs=[r[0] for r in extras]
   elif kind=='place_hold_retract':cs=[c for c in [.3,.4,.5] for h in [.81,.84,.87]];extras=[h for c in [.3,.4,.5] for h in [.81,.84,.87]]
   elif kind=='back_departure':cs=[.3,.35,.4]
   elif kind=='side_departure':cs=[.4,.55,.7]
   elif kind=='turn_endpoint':cs=np.radians([90,135,180]).tolist()
   else:extras=[[.4,.8,0,0,1,1],[.8,.4,-.2,.2,1,1],[.6,.6,.3,.3,0,1],[.3,.9,0,0,1,0]];cs=[0]*len(extras)
   seed=seedbase+100*pi+si;raw,src,c=generate(m,kind,cs,pi,[seed]*len(cs),extras)
   with torch.no_grad():pos,poses,root=fk(raw,canonical=False,return_pose=True);q=prev.pa.core.quantity(fk(raw),src['task'],HH/fk.height)
   for j,cmd in enumerate(cs):
    s=f'{kind}_p{pi}_s{si}';path=out/(s+f'_c{j}.npz');np.savez_compressed(path,motion=raw[j].numpy(),joints_zup_m=pos[j].numpy(),poses_axisangle=poses[j].numpy(),root_translation=root[j].numpy(),human_height=fk.height,human_quantity=float(q[j]),fps=20.,task=src['task'],command=cmd,control_features=c[j].numpy());rows.append(dict(kind=kind,task=src['task'],source=s,seed=seed,command=cmd,command_index=j,extra=None if extras is None else extras[j],path=str(path),generator=weight,generator_sha256=weight_hash,split=a.split))
   (out/'manifest.json').write_text(json.dumps(rows,indent=2));print(len(rows),kind,round(time.time()-start,1),flush=True)
(out/'complete.json').write_text(json.dumps(dict(requests=len(rows),elapsed_s=time.time()-start)))
