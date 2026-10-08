"""Generate new classes from official training annotations and fresh noise only.
No source motion tensor is read. Exact official part language avoids conflicting
hand-authored descriptions. These are exploratory development candidates.
"""
import sys,json,hashlib
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/universal_tracker_20261006';sys.path.insert(0,str(R/'outputs_amass/franken_eleven_20261003/code'))
from core import torch,np,FK,load_model,sample,OLD
from hydra.utils import instantiate
from omegaconf import OmegaConf
from src.data.text_part_utils import load_from_annotation_with_model
from src.data.text_motion import align
out=D/'generated_point_corrected';out.mkdir(exist_ok=False);(out/'prompts').mkdir();(out/'human').mkdir();torch.set_num_threads(4)
annotations_path=R/'datasets/annotations/frankenstein-dataset/annotations/annotations.json';annotations=json.loads(annotations_path.read_text());inventory=json.loads((D/'action_inventory_v2.json').read_text());cfg=OmegaConf.load(OLD/'base_config.yaml');enc=instantiate(cfg.data.text_encoder);enc.no_model=False;enc.rand_mask=False;prompts=[]
for category,items in {'point':[r for r in inventory['candidate_manifest']['train']['point'] if r['uid'] in ('01584','01036')]}.items():
 for item in items[:2]:
  row=annotations[item['uid']];n=int(20*(row['end']-row['start']));emb,_=load_from_annotation_with_model(enc,row['annotations'],row['path'],row['start'],row['end']);local=align(torch.zeros(n,205),emb['local']['x']);tx={'x':emb['x'][None],'length':torch.tensor([emb['length']])};path=out/'prompts'/f'{category}_{item["uid"]}.json';path.write_text(json.dumps(row,indent=2));torch.save(dict(local=local,tx=tx),path.with_suffix('.pt'));prompts.append(dict(task=category,uid=item['uid'],frames=n,prompt=str(path),caption=row['caption_label']))
del enc
model,_=load_model('cpu');fk=FK('cpu');rows=[]
for item in prompts:
 z=torch.load(Path(item['prompt']).with_suffix('.pt'),map_location='cpu',weights_only=False)
 for si in range(2):
  seed=97062000+si;raw=sample(model,z['local'][None],z['tx'],0,torch.zeros(1),[seed],False)
  with torch.no_grad():pos,poses,root=fk(raw,canonical=False,return_pose=True)
  path=out/'human'/f'{item["task"]}_{item["uid"]}_s{si}.npz';np.savez_compressed(path,motion=raw[0].numpy(),joints_zup_m=pos[0].numpy(),poses_axisangle=poses[0].numpy(),root_translation=root[0].numpy(),human_height=fk.height,fps=20.)
  rows.append(dict(item,source=item['task']+'_'+item['uid']+f'_s{si}',seed=seed,command=None,path=str(path),source_type='FrankenMotion_generated_from_official_train_text_only',split='development',human_sha256=hashlib.sha256(path.read_bytes()).hexdigest()))
  (out/'manifest.json').write_text(json.dumps(rows,indent=2));print(len(rows),item['task'],item['uid'],flush=True)
(out/'protocol.json').write_text(json.dumps(dict(annotation_sha256=hashlib.sha256(annotations_path.read_bytes()).hexdigest(),scope='Exact official training text annotations and independent diffusion noise. No source motion loaded; no numeric command and no output editing. Exploratory generated demos, not a held-out semantic test.',rows=len(rows),sampler='50-step DDIM, guidance1, CPU',generator_checkpoint=str(R/'outputs_amass/official_20260916/logs/checkpoints/last.ckpt')),indent=2))
