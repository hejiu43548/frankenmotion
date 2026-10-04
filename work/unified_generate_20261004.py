"""Generate disjoint source seeds with the immutable integrated V3 generator."""
import sys,json,argparse,hashlib
from pathlib import Path
R=Path('/home/pku/frankenmotion');sys.path.insert(0,str(R/'work'))
import physical_adapter_20261003 as pa
from core import torch,np,FK,sample
from generate import inputs
N=pa.OUT;U=N.parent/'franken_unified_20261004'
p=argparse.ArgumentParser();p.add_argument('--split',choices=['development_validation','training_extra','final_test'],required=True);a=p.parse_args()
if a.split=='final_test':
 selection=json.loads((U/'frozen_unified/protocol.json').read_text());assert hashlib.sha256(Path(selection['checkpoint']).read_bytes()).hexdigest()==selection['checkpoint_sha256'] and not selection['task_routing']
root=U/a.split;root.mkdir(exist_ok=False);out=root/'human';out.mkdir()
protocol=json.loads((U/'protocol.json').read_text());base={'development_validation':88041000,'training_extra':92041000,'final_test':94041000}[a.split];nseeds=4 if a.split=='final_test' else 2
frozen=json.loads((N/'frozen_integrated_v3/protocol.json').read_text());sources=json.loads((pa.BASE/'evaluation_manifest.json').read_text());sources=[r for r in sources if int(r['source'].split('_s')[-1])<nseeds and (a.split!='development_validation' or '_p0_' in r['source'])]
(root/'protocol.json').write_text(json.dumps(dict(split=a.split,seed_base=base,sources=len(sources),generation=frozen['generation'],weight_hashes=frozen['weight_hashes'],purpose='Development choice only' if a.split=='development_validation' else a.split),indent=2))
torch.set_num_threads(2);torch.cuda.set_per_process_memory_fraction(.15);fk=FK('cuda');model=None;current=None;rows=[]
for src in sources:
 weight=N/frozen['generation'].get(src['task'],frozen['generation']['all_other_tasks']);assert hashlib.sha256(weight.read_bytes()).hexdigest()==frozen['weight_hashes'][str(weight)]
 if weight!=current:
  if model is not None:del model;torch.cuda.empty_cache()
  model,_=pa.core.load_model(task_weights=weight);current=weight
 pi=int(src['source'].split('_p')[-1].split('_')[0]);si=int(src['source'].split('_s')[-1]);seed=base+pi*100+si;cs=src['commands'];local,tx,cmd,controls=inputs(src,cs);raw=sample(model,local,tx,src['task_id'],cmd,[seed]*len(cs),True,root_controls=controls)
 with torch.no_grad():pos,poses,rootpos=fk(raw,canonical=False,return_pose=True)
 for i,c in enumerate(cs):
  path=out/(src['source']+f'_c{i}.npz');np.savez_compressed(path,motion=raw[i].cpu().numpy(),joints_zup_m=pos[i].cpu().numpy(),poses_axisangle=poses[i].cpu().numpy(),root_translation=rootpos[i].cpu().numpy(),human_height=fk.height,fps=20.,task=src['task'],command=c)
  rows.append(dict(task=src['task'],source=src['source'],seed=seed,command=c,command_index=i,path=str(path),generator=str(weight),generator_sha256=frozen['weight_hashes'][str(weight)]))
 print(len(rows),'generated',flush=True)
(root/'manifest.json').write_text(json.dumps(rows,indent=2))
