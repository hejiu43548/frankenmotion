from uc_common import *
from unified_control import SharedCommands
from legacy_targets import LegacyTargets
from conditions import random_batch,KINDS
import argparse,time,hashlib
p=argparse.ArgumentParser();p.add_argument('--name',required=True);p.add_argument('--width',type=int,default=512);p.add_argument('--steps',type=int,default=20000);p.add_argument('--lr',type=float,default=1e-4);p.add_argument('--initial');p.add_argument('--boundaries',action='store_true');p.add_argument('--seed',type=int,default=107072222);p.add_argument('--save-every',type=int,default=1000);a=p.parse_args();out=D/'training'/a.name;out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.14);torch.manual_seed(a.seed);teacher=LegacyTargets().cuda();student=SharedCommands(a.width).cuda()
if a.initial:student.load_state_dict(torch.load(a.initial,map_location='cuda',weights_only=False)['controller'])
opt=torch.optim.AdamW(student.parameters(),lr=a.lr,weight_decay=1e-6);vg=torch.Generator(device='cuda').manual_seed(107072223);vc,groups=random_batch(128,generator=vg);vr,vo=teacher(vc);scale=vr.std(0).clamp_min(.02);oscale=vo[groups>=13].std(0).clamp_min(.02);start=time.monotonic()
(out/'protocol.json').write_text(json.dumps(dict(kind='Direct shared conditional residual distillation',fields=FIELDS,intents=INTENTS,skills=KINDS,width=a.width,steps=a.steps,lr=a.lr,initial=a.initial,trainable_parameters=sum(v.numel() for v in student.parameters()),scope='One dense shared encoder and4 layer heads plus shared205-output head. Heads are injection locations, not task-specific experts. All parameters used for all inputs. Frozen training-only legacy branches.',boundary_augmentation=a.boundaries,seed=a.seed,sampled='Fresh continuous commands and normalized phases every update, balanced17 semantic contexts. Fixed independent feature-validation RNG. This is not motion-level validation.',loss='Layer residual sum and per-denoising-step output residual, variance normalized. No post-sampling edits.'),indent=2))
def save(step):
 f=out/f'model_{step:05d}.pt';tmp=f.with_suffix('.tmp');torch.save(dict(controller={k:v.detach().cpu() for k,v in student.state_dict().items()},width=a.width,step=step,optimizer=opt.state_dict(),fields=FIELDS,intents=INTENTS),tmp);tmp.replace(f);f.with_suffix('.ready.json').write_text(json.dumps(dict(sha256=hashlib.sha256(f.read_bytes()).hexdigest())))
def validate(step):
 with torch.no_grad():
  r,o=student(vc);errors=[dict(kind=k,residual_rmse=float((r[groups==i]-vr[groups==i]).square().mean().sqrt()),output_rmse=float((o[groups==i]-vo[groups==i]).square().mean().sqrt())) for i,k in enumerate(KINDS)]
 status=dict(step=step,total=a.steps,elapsed_s=time.monotonic()-start,errors=errors);(out/'status.json').write_text(json.dumps(status,indent=2));(out/f'validation_{step:05d}.json').write_text(json.dumps(status,indent=2));print(json.dumps(dict(step=step,elapsed_s=status['elapsed_s'],mean_r=sum(r['residual_rmse'] for r in errors)/len(errors),max_r=max(r['residual_rmse'] for r in errors),mean_o=sum(r['output_rmse'] for r in errors)/len(errors))),flush=True);save(step)
validate(0)
for step in range(1,a.steps+1):
 c,g=random_batch(16,boundaries=a.boundaries);r,o=teacher(c);pr,po=student(c);loss=((pr-r)/scale).square().mean()+((po-o)/oscale).square().mean();assert torch.isfinite(loss);opt.zero_grad();loss.backward();torch.nn.utils.clip_grad_norm_(student.parameters(),2);opt.step()
 if step%a.save_every==0 or step==a.steps:validate(step)
(out/'complete.json').write_text(json.dumps(dict(steps=a.steps,elapsed_s=time.monotonic()-start)))
