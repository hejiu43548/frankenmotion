"""Frozen candidate, fresh noise, two off-grid commands per class; no optimization."""
from common import *
from deployment_runtime import load_one_checkpoint
import hashlib
out=D/'interpolation_probe';out.mkdir(exist_ok=False);torch.set_num_threads(2);fk=FK('cpu');ck=D/'exports/selected_final/unified_generator.pt';student=load_one_checkpoint(ck);rows=[]
for tid,task in enumerate(TASKS):
 src=next(r for r in SOURCES if r['task']==task and '_p3_' in r['source']);lo,hi=RANGES[tid];cs=[lo+.3*(hi-lo),lo+.7*(hi-lo)];local,tx,cmd,controls=inputs(src,cs,'cpu',fk);teacher,_=pa.core.load_model('cpu',task_weights=teacher_path(task))
 for name,m in [('old',teacher),('unified',student)]:
  raw=sample(m,local,tx,tid,cmd,[107081000]*2,True,root_controls=controls)
  with torch.no_grad():pos,poses,root=fk(raw,canonical=False,return_pose=True);q=pa.core.quantity(fk(raw),tid,HH/fk.height)
  for i,c in enumerate(cs):
   p=out/(task+f'_{name}_c{i}.npz');np.savez_compressed(p,motion=raw[i].numpy(),joints_zup_m=pos[i].numpy(),poses_axisangle=poses[i].numpy(),root_translation=root[i].numpy(),human_height=fk.height,human_quantity=float(q[i]),fps=20.,task=task,command=c);rows.append(dict(task=task,variant=name,command=c,path=str(p),source=task+'_p3_s0',command_index=i))
 del teacher
(out/'manifest.json').write_text(json.dumps(rows,indent=2));(out/'protocol.json').write_text(json.dumps(dict(checkpoint=str(ck),sha256=hashlib.sha256(ck.read_bytes()).hexdigest(),seed=107081000,commands='30% and70% of task range, absent from five-level distillation grid',purpose='Small interpolation probe, not broad language generalization or physical evaluation.'),indent=2))
