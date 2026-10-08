"""Inventory available source motions without treating overlapping crops as held-out."""
from pathlib import Path
from collections import Counter, defaultdict
import json,re,hashlib

R=Path('/home/pku/frankenmotion'); D=R/'outputs_amass/universal_tracker_20261006'
A=R/'datasets/annotations/frankenstein-dataset/annotations'
rows=json.loads((A/'annotations.json').read_text())
split_ids={s:set((A/'splits'/f'{s}.txt').read_text().split()) for s in ('train','val','test')}
source_splits=defaultdict(set)
for s,ids in split_ids.items():
    for uid in ids:
        if uid in rows: source_splits[rows[uid]['path']].add(s)
patterns={
 'run':r'\brun(?:ning)?\b|\bjog(?:ging)?\b',
 'squat':r'\bsquat\w*|\bcrouch\w*',
 'stretch':r'\bstretch\w*',
 'dance':r'\bdanc\w*',
 'punch':r'\bpunch\w*|\bboxing\b',
 'throw':r'\bthrow\w*',
 'clap':r'\bclap\w*',
 'point':r'\bpoint(?:s|ing)?\b',
 'balance':r'\bbalanc\w*|stand(?:s|ing)? on (?:one|a single) leg',
 'lunge':r'\blung(?:e|es|ing)\b',
 'bend':r'\bbend\w*|\bbow(?:s|ing)?\b',
 'sidestep':r'\bside.?step\w*|\bsideways\b',
}
catalog={s:{k:[] for k in patterns} for s in split_ids}; available=Counter(); ambiguity=0
for uid,row in rows.items():
    p=R/'datasets/motions/AMASS_20.0_fps_nh_smplrifke'/(row['path']+'.npy')
    if not p.exists(): continue
    sset=source_splits[row['path']]
    if len(sset)!=1: ambiguity+=1;continue
    split=next(iter(sset));available[split]+=1
    labels=[a['text'] for a in row.get('annotations',[]) if a.get('bodypart') in ('action','sequence_caption')]
    caption=row.get('caption_label','');text=' '.join(labels+[caption]).lower()
    for category,pattern in patterns.items():
        if re.search(pattern,text) and 1.5<=row['duration']<=15:
            catalog[split][category].append(dict(uid=uid,source=row['path'],motion_path=str(p),start=row['start'],end=row['end'],duration=row['duration'],caption=caption,action_labels=labels))
counts={s:{k:len({r['source'] for r in v}) for k,v in c.items()} for s,c in catalog.items()}
selected={s:{} for s in catalog}
for s,c in catalog.items():
    for k,rs in c.items():
        seen=set();take=[]
        for row in sorted(rs,key=lambda r:hashlib.sha256(('6106'+r['uid']).encode()).hexdigest()):
            if row['source'] in seen:continue
            seen.add(row['source']);take.append(row)
            if len(take)>=(20 if s=='train' else 5):break
        selected[s][k]=take
babel=json.loads(Path('/home/pku/datasets/motion_compare/babel/babel_v1.0_release/train.json').read_text());bc=Counter()
for row in babel.values():
    for a in (row.get('frame_ann') or row.get('seq_ann') or {}).get('labels',[]):bc.update(a.get('act_cat') or [])
result=dict(annotation_source=str(A/'annotations.json'),annotation_sha256=hashlib.sha256((A/'annotations.json').read_bytes()).hexdigest(),available_unambiguous=dict(available),excluded_crops_with_source_split_overlap=ambiguity,unique_source_counts=counts,babel_train_action_counts=dict(bc.most_common()),candidate_manifest=selected,scope='Candidate inventory only, not a claim of trained classes or unseen generator data. Generator was pretrained on official training split; tracker split is source-disjoint here.')
(D/'action_inventory.json').write_text(json.dumps(result,indent=2,ensure_ascii=False));print(json.dumps({k:result[k] for k in ['available_unambiguous','excluded_crops_with_source_split_overlap','unique_source_counts']},indent=2))
