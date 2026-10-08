from common import *
import time,hashlib
out=D/'teacher_training';out.mkdir(exist_ok=False);torch.set_num_threads(4);fk=FK('cpu');model=None;current=None;rows=[];start=time.time()
for src in SOURCES:
 pi=int(src['source'].split('_p')[1].split('_')[0])
 if pi not in [0,1]:continue
 weight=teacher_path(src['task'])
 if current!=weight:
  if model is not None:del model
  model,_=pa.core.load_model('cpu',task_weights=weight);current=weight
 for noise_i in range(3):
  seed=107072000+pi*100+noise_i;lo,hi=RANGES[src['task_id']];cs=np.linspace(lo,hi,5).tolist();local,tx,cmd,controls=inputs(src,cs,'cpu',fk);raw=sample(model,local,tx,src['task_id'],cmd,[seed]*5,True,root_controls=controls)
  with torch.no_grad():pos,poses,root=fk(raw,canonical=False,return_pose=True)
  for k,c in enumerate(cs):
   f=out/(src['task']+f'_p{pi}_s{noise_i}_c{k}.npz');np.savez_compressed(f,motion=raw[k].numpy(),joints_zup_m=pos[k].numpy(),poses_axisangle=poses[k].numpy(),root_translation=root[k].numpy(),fps=20.,human_height=fk.height,command=c,task=src['task']);rows.append(dict(task=src['task'],task_id=src['task_id'],prompt=src['prompt'],frames=src['frames'],caption=src['caption'],source=src['source'],seed=seed,command=c,path=str(f),teacher=str(weight),sha256=hashlib.sha256(f.read_bytes()).hexdigest()))
  (out/'manifest.json').write_text(json.dumps(rows,indent=2));print(len(rows),src['task'],round(time.time()-start,1),flush=True)
assert len(rows)==330
(out/'complete.json').write_text(json.dumps(dict(count=len(rows),elapsed_s=time.time()-start)))
