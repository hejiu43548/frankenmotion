"""Additional training diffusion noise; never used as final evaluation."""
import sys,json,hashlib,datetime
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/jump_tracker_20261007';sys.path[:0]=[str(R/'work'),str(R/'outputs_amass/franken_eleven_20261003/code')]
import physical_adapter_20261003 as pa
from core import torch,np,FK,sample,encode_control,HH

out=D/'expanded_training';out.mkdir(exist_ok=False);(out/'human').mkdir();torch.set_num_threads(4);fk=FK('cpu');model=None;current=None;rows=[]
frozen=json.loads((pa.OUT/'frozen_integrated_v3/protocol.json').read_text());sources=json.loads((pa.BASE/'evaluation_manifest.json').read_text());sources=[r for r in sources if r['task']=='jump' and int(r['source'].split('_s')[-1])<2];assert len(sources)==8
protocol=dict(split='expanded_training',seed_base=107070000,generation=frozen['generation'],weight_hashes=frozen['weight_hashes'],started_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),scope='Training-only additional40 diffusion samples, same prompt templates, different seeds from development and predeclared final seed107071000. Frozen heterogeneous generator adapters and frozen GMR refinement unchanged across trackers. CPU generation,50 DDIM steps.')
(out/'protocol.json').write_text(json.dumps(protocol,indent=2))
for src in sources:
 weight=pa.OUT/frozen['generation'].get(src['task'],frozen['generation']['all_other_tasks']);assert hashlib.sha256(weight.read_bytes()).hexdigest()==frozen['weight_hashes'][str(weight)]
 if weight!=current:
  if model is not None:del model
  model,_=pa.core.load_model('cpu',task_weights=weight);current=weight
 pi=int(src['source'].split('_p')[-1].split('_')[0]);si=int(src['source'].split('_s')[-1]);seed=protocol['seed_base']+pi*100+si;cs=list(src['commands']);z=torch.load(Path(src['prompt']).with_suffix('.pt'),map_location='cpu',weights_only=False);b=len(cs);local=z['local'][None].expand(b,-1,-1);tx={k:v.repeat((b,)+(1,)*(v.ndim-1)) if torch.is_tensor(v) else v for k,v in z['tx'].items()};cmd=torch.tensor(cs,dtype=torch.float32);controls=None
 if src['task'] in ['walk','back_walk','turn']:
  values=torch.zeros(b,src['frames'],2)
  if src['task']=='turn':values[...,1]=-cmd[:,None]/((src['frames']-1)/20)
  else:values[...,0]=cmd[:,None]*fk.height/HH
  controls=encode_control(values,torch.ones(b,src['frames'],dtype=torch.bool))
 raw=sample(model,local,tx,src['task_id'],cmd,[seed]*b,True,root_controls=controls)
 with torch.no_grad():pos,poses,rootpos=fk(raw,canonical=False,return_pose=True)
 for i,c in enumerate(cs):
  path=out/'human'/('expanded_'+src['source']+f'_c{i}.npz');np.savez_compressed(path,motion=raw[i].numpy(),joints_zup_m=pos[i].numpy(),poses_axisangle=poses[i].numpy(),root_translation=rootpos[i].numpy(),human_height=fk.height,fps=20.,task=src['task'],command=c);rows.append(dict(task=src['task'],source=src['source'],seed=seed,command=c,command_index=i,command_set=('standard_grid' if i<5 else 'interpolation'),path=str(path),generator=str(weight),generator_sha256=frozen['weight_hashes'][str(weight)],human_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
 (out/'manifest.json').write_text(json.dumps(rows,indent=2));print(len(rows),'generated',flush=True)
assert len(rows)==40
