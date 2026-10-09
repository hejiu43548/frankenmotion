"""Replace only turn with event crops; preserve every other task cache byte."""
from pathlib import Path
import argparse,sys,importlib.util
from common import *

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
 old=ROOT/'outputs_amass/unified_direct_20261008/data';selection=Path('/mnt/sda2/frankenmotion/outputs_amass/walking_turn_admission_20261009_v2')
 anns=json.loads((ROOT/'datasets/annotations/frankenstein-dataset/annotations/annotations.json').read_text())
 skpath=ROOT/'outputs_amass/transfer_charlie_20261008/snapshot/outputs_amass/franken_eleven_20261003/skeleton.npz';sk=Skeleton(skpath);torch.set_num_threads(2)
 spec=importlib.util.spec_from_file_location('source_conversion',ROOT/'work/unified_direct/prepare_data.py');conv=importlib.util.module_from_spec(spec);spec.loader.exec_module(conv)
 from src.data.text_part import TextEmbeddings
 cfg=json.loads((ROOT/'pretrained/official/config.json').read_text())['data']['text_encoder'].copy();cfg.pop('_target_');cfg['rand_mask']=False;text=TextEmbeddings(**cfg)
 pca=np.load(old/'pca.npz');text.pca_components.mean_=pca['mean'].copy();text.pca_components.components_=pca['components'].copy()
 np.savez(a.out/'pca.npz',mean=pca['mean'],components=pca['components'])
 reduced={}
 def embed(label):
  e=text.get_embedding(label)['x'];assert e.shape==(1,512)
  if label not in reduced:reduced[label]=text.pca_components.transform(e.cpu().numpy())[0]
  return e.cpu().numpy()[0],reduced[label]
 source_hashes={};speeds=[];report={};(a.out/'cache').mkdir()
 for split in ['train','val']:
  original=json.loads((old/f'{split}.json').read_text());keep=[r for r in original if r['task_id']!=4];new=[]
  rows=json.loads((selection/f'turn_{split}.json').read_text())
  for r in rows:
   full=old/'motions'/(r['family']+'.npy')
   if not full.exists():
    full=a.out/'motions'/(r['family']+'.npy');status=conv.convert((r['family'],r['raw_path'],str(full),str(skpath)));assert status['status'] in ['ok','cached'],status
   if str(full) not in source_hashes:source_hashes[str(full)]=sha(full)
   s,e=r['crop_start_frame_20fps'],r['crop_end_frame_20fps'];motion=np.load(full)[s:e].copy();n=e-s;assert len(motion)==n
   ann=anns[r['key']];runs=text._format_annotations(ann['annotations'],s/20,e/20,20)
   # Explicit integer frame times avoid float-duration truncation in the official formatter.
   times=np.arange(s,e)/20;local=np.zeros((n,408),np.float32);mask=np.zeros_like(local,dtype=bool)
   cropped=[]
   for part_idx,part in enumerate(text.body_part_order):
    labels=[]
    for t in times:
     label=next((x['text'] for x in ann['annotations'] if x['bodypart']==part and x['start']<=t<x['end']),'unknown');labels.append(label)
    for i,label in enumerate(labels):
     if label!='unknown':local[i,part_idx*51:(part_idx+1)*51]=embed(label)[1];mask[i,part_idx*51:(part_idx+1)*51]=True
   for x in ann['annotations']:
    lo=max(s/20,x['start']);hi=min(e/20,x['end'])
    if hi>lo:cropped.append(dict(x,start=lo-s/20,end=hi-s/20))
   # Preserve user-selected AMASS global text; retain full original caption provenance.
   tx,_=embed(ann['caption_label']);q=float(quantity(sk(torch.from_numpy(motion)[None]),4,1)[0]);assert np.isfinite(q) and q*r['metrics']['body_turn_rad']>0
   cache=a.out/'cache'/f'{split}_turn_{r["id"]}.npz';np.savez(cache,motion=motion,local=local,local_mask=mask,tx=tx,quantity=np.float32(q),task=np.int64(4))
   item=dict(r,annotation_key=r['key'],key='turn_event_'+r['id'],cache=str(cache),task_id=4,target_frames=n,quantity=q,real_frames=n,pad_frames=0,ready_for_training=True,cropped_annotations=cropped,text_policy='Original AMASS global caption; part/action/trajectory embeddings recropped on absolute source time; not synthetic templates.',motion_source=str(full),cache_sha256=sha(cache))
   new.append(item)
   if split=='train':speeds.append(float(np.linalg.norm(motion[:-1,1:3],axis=-1).mean()*20))
  allrows=keep+new;assert len(allrows)==len(original)-sum(r['task_id']==4 for r in original)+len(rows)
  save_json(a.out/f'{split}.json',allrows);report[split]=dict(total=len(allrows),new_turn=len(new),unchanged_other_tasks=len(keep),original_manifest_sha256=sha(old/f'{split}.json'),selection_sha256=sha(selection/f'turn_{split}.json'))
 train=json.loads((a.out/'train.json').read_text());val=json.loads((a.out/'val.json').read_text());assert not {r['family'] for r in train}&{r['family'] for r in val}
 policy=dict(native_speed_m_s=float(np.median(speeds)),human_equivalent_speed_m_s=float(np.median(speeds))*HUMAN_EQUIVALENT_HEIGHT/sk.height,source='median mean horizontal speed over real frames of new TRAIN turn crops only',turn_training_records=len(speeds),task='walking turn, fixed speed and requested signed angle',range_rad=[-3.5,3.5],stage3_command_magnitude_range_rad=[.35,3.5],stage3_direction='match measured turn direction of training text source',spin_added=False)
 save_json(a.out/'turn_policy.json',policy);save_json(a.out/'source_sha256.json',source_hashes);save_json(a.out/'data_audit.json',dict(verified=True,by_split=report,source_family_split_disjoint=True,all_other_rows_preserved_exactly=True,new_turn_no_padding=True,turn_labels_recomputed_on_actual_crop=True,prepare_source_sha256=sha(__file__),pca_source_sha256=sha(old/'pca.npz'),pca_parameters_reused_exactly=True))
 print(json.dumps(dict(report=report,turn_policy=policy),indent=2))
if __name__=='__main__':main()
