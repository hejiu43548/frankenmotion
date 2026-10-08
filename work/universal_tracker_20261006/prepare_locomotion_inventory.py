"""Add natural walk/jump/turn coverage without looking at validation outcomes."""
from pathlib import Path
import json,re,hashlib
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/universal_tracker_20261006';A=R/'datasets/annotations/frankenstein-dataset/annotations';rows=json.loads((A/'annotations.json').read_text());splits={s:set((A/'splits'/(s+'.txt')).read_text().split()) for s in ['train','val']};old=json.loads((D/'natural_extended/manifest.json').read_text());used={s:{r['source'] for r in old if r['split']==s} for s in splits};patterns={'walk':r'\bwalk(?:s|ing)?\b','jump':r'\bjump\w*|\bhop(?:s|ping)?\b|\bleap\w*','turn':r'\bturn(?:s|ing)?\b|\brotat\w*'};result={'candidate_manifest':{s:{} for s in splits}}
for split,ids in splits.items():
 for task,pattern in patterns.items():
  candidates=[]
  for uid in ids:
   if uid not in rows:continue
   row=rows[uid];path=R/'datasets/motions/AMASS_20.0_fps_nh_smplrifke'/(row['path']+'.npy');text=' '.join([row.get('caption_label','')]+[r['text'] for r in row.get('annotations',[]) if r.get('bodypart')=='action']).lower()
   if row['path'] not in used[split] and 1.5<=row['duration']<=12 and path.exists() and re.search(pattern,text):candidates.append(dict(uid=uid,source=row['path'],motion_path=str(path),start=row['start'],end=row['end'],duration=row['duration'],caption=row.get('caption_label','')))
  selected=[]
  for row in sorted(candidates,key=lambda r:hashlib.sha256(('6106loc'+r['uid']).encode()).hexdigest()):
   if row['source'] in used[split]:continue
   used[split].add(row['source']);selected.append(row)
   if len(selected)>=(30 if split=='train' else 5):break
  result['candidate_manifest'][split][task]=selected
train={r['source'] for v in result['candidate_manifest']['train'].values() for r in v};val={r['source'] for v in result['candidate_manifest']['val'].values() for r in v};assert not train&val;assert not train&used['val'];assert not val&used['train'];result['scope']='Additional natural locomotion and ballistic references; deterministic hash sampling, training sources distinct from validation and prior natural set.';(D/'locomotion_inventory.json').write_text(json.dumps(result,indent=2));print({s:{k:len(v) for k,v in c.items()} for s,c in result['candidate_manifest'].items()})
