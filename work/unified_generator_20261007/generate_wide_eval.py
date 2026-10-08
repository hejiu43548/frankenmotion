from common import *
from wide_adapter import enable
enable()
import argparse,time,hashlib
p=argparse.ArgumentParser();p.add_argument('--checkpoint');p.add_argument('--name',required=True);p.add_argument('--split',choices=['development','final'],default='development');p.add_argument('--teachers',action='store_true');a=p.parse_args();assert bool(a.checkpoint)!=a.teachers
out=D/'generation'/a.name;out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(3);fk=FK('cpu');model=None;current=None;rows=[];start=time.time();checkpoint=Path(a.checkpoint) if a.checkpoint else None
(out/'protocol.json').write_text(json.dumps(dict(split=a.split,checkpoint=str(checkpoint),checkpoint_sha256=hashlib.sha256(checkpoint.read_bytes()).hexdigest() if checkpoint else None,teachers=a.teachers,steps=50,seed_base=107073000 if a.split=='development' else 107074000,scope='Fresh diffusion noise; fixed existing prompt templates, not unseen-text generalization. Development uses p0 andp2; final all4 prompts,4 noises each. One shared checkpoint for every task unless explicitly teacher baseline.'),indent=2))
for src in SOURCES:
 pi=int(src['source'].split('_p')[1].split('_')[0])
 if a.split=='development' and pi not in [0,2]:continue
 weight=teacher_path(src['task']) if a.teachers else checkpoint
 if current!=weight:
  if model is not None:del model
  model,_=pa.core.load_model('cpu',task_weights=weight);current=weight
 for si in range(1 if a.split=='development' else 4):
  seed=(107073000 if a.split=='development' else 107074000)+100*pi+si;cs=src['commands'];local,tx,cmd,controls=inputs(src,cs,'cpu',fk);raw=sample(model,local,tx,src['task_id'],cmd,[seed]*len(cs),True,root_controls=controls)
  with torch.no_grad():pos,poses,root=fk(raw,canonical=False,return_pose=True);q=pa.core.quantity(fk(raw),src['task'],HH/fk.height)
  for i,c in enumerate(cs):
   source=f'{src["task"]}_p{pi}_s{si}';path=out/(source+f'_c{i}.npz');np.savez_compressed(path,motion=raw[i].numpy(),joints_zup_m=pos[i].numpy(),poses_axisangle=poses[i].numpy(),root_translation=root[i].numpy(),human_height=fk.height,human_quantity=float(q[i]),fps=20.,task=src['task'],command=c);rows.append(dict(task=src['task'],source=source,seed=seed,command=c,command_index=i,path=str(path),generator=str(weight),generator_sha256=hashlib.sha256(weight.read_bytes()).hexdigest(),human_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
  (out/'manifest.json').write_text(json.dumps(rows,indent=2));print(len(rows),src['task'],round(time.time()-start,1),flush=True)
expected=110 if a.split=='development' else 880;assert len(rows)==expected
(out/'complete.json').write_text(json.dumps(dict(requests=len(rows),elapsed_s=time.time()-start)))
