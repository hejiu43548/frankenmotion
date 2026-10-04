import sys,json,argparse,hashlib
from pathlib import Path
sys.path.insert(0,'/home/pku/frankenmotion/work')
import physical_adapter_20261003 as pa
from core import torch,np,FK,TASKS,sample
from generate import inputs
BASE=pa.BASE;OUT=pa.OUT
if __name__=='__main__':
 ap=argparse.ArgumentParser();ap.add_argument('--phase',choices=['development','confirmation'],default='development');ap.add_argument('--weights',default=str(OUT/'physical_adapter/best.pt'));ap.add_argument('--label',default='physical');ap.add_argument('--tasks',default='');a=ap.parse_args();folder=OUT/'generated'/(a.label+'_'+a.phase);folder.mkdir(parents=True,exist_ok=True)
 (folder/'provenance.json').write_text(json.dumps(dict(weights=a.weights,sha256=hashlib.sha256(Path(a.weights).read_bytes()).hexdigest(),phase=a.phase,label=a.label,sampler='DDIM 50 eta0 guidance1',confirmation_noise='42026000+prompt*100+seed'),indent=2))
 torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.15);m,_=pa.core.load_model(task_weights=Path(a.weights));fk=FK('cuda');rows=json.loads((BASE/'evaluation_manifest.json').read_text());result=[]
 if a.tasks:rows=[r for r in rows if r['task'] in a.tasks.split(',')]
 if a.phase=='development':rows=[r for r in rows if r['source'].endswith('_p0_s0')]
 for src in rows:
  tid=src['task_id'];pi=int(src['source'].split('_p')[-1].split('_')[0]);si=int(src['source'].split('_s')[-1]);seed=990700+tid if a.phase=='development' else 42026000+pi*100+si
  cs=src['commands'];local,tx,cmd,controls=inputs(src,cs);raw=sample(m,local,tx,tid,cmd,[seed]*len(cs),True,root_controls=controls)
  with torch.no_grad():pos,poses,root=fk(raw,canonical=False,return_pose=True);q=pa.core.quantity(fk(raw),tid,pa.HH/fk.height)
  for i,c in enumerate(cs):
   path=folder/(src['source']+f'_c{i}.npz');np.savez_compressed(path,motion=raw[i].cpu().numpy(),joints_zup_m=pos[i].cpu().numpy(),poses_axisangle=poses[i].cpu().numpy(),root_translation=root[i].cpu().numpy(),human_height=fk.height,human_quantity=float(q[i]),fps=20.,task=src['task'],command=c)
   result.append(dict(task=src['task'],source=src['source'],seed=seed,command=c,command_index=i,path=str(path)))
  print(src['source'],[round(float(x),3) for x in q],flush=True)
 (OUT/f'{a.label}_{a.phase}_manifest.json').write_text(json.dumps(result,indent=2))
