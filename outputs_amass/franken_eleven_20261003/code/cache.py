import re,hashlib,json
from collections import Counter
from core import *
from omegaconf import OmegaConf
from hydra.utils import instantiate
import src.prepare
from src.tools.inference import load_smplh
from src.tools.smplrifke_feats import smplrifkefeats_to_smpldata

PATTERNS=[r'(rais|lift|reach).*(hand|arm|up)',r'reach|point|extend.*arm',r'\bpunch|\bbox|\bjab',r'\bwav',r'\bturn|\brotat|\bspin',r'side.?step|sideways|side to side',r'walk.*back|step.*back|backward.*walk',r'\bkick',r'\bjump|\bhop',r'\blean|bend.*forward|\bbow\b',r'\bwalk']
def main():
    torch.set_num_threads(2);OUT.mkdir(exist_ok=True);(OUT/'cache').mkdir(exist_ok=True)
    smpl=load_smplh();J=(smpl.smplh.J_regressor@smpl.smplh.v_template).detach().numpy();parents=smpl.smplh.parents.cpu().numpy();height=abs((J[16,1]+J[17,1]-J[7,1]-J[8,1])/2)
    np.savez(OUT/'skeleton.npz',J=J,parents=parents,height=height)
    fk=FK();cfg=OmegaConf.load(OLD/'base_config.yaml');cfg.data.preload=False;cfg.data.text_encoder.preload=True;cfg.data.text_encoder.rand_mask=False;cfg.data.drop_cond=0.;cfg.data.drop_trans=0.
    directory=ROOT/'datasets/annotations/frankenstein-local-available/annotations';ann=json.loads((directory/'annotations.json').read_text())
    vals=(directory/'splits/val.txt').read_text().split();valfamilies={ann[k]['path'] for k in vals if k in ann}
    records=[];parity=[]
    for split,limit in [('train',128),('val',16)]:
        ds=instantiate(cfg.data,split=split);ds.is_training=False
        for task,pattern in enumerate(PATTERNS):
            candidates=[k for k in ds.keyids if re.search(pattern,ann[k].get('caption_label',''),re.I) and not re.search(r'crawl|cartwheel|lie down|lying|sit on|somersault',ann[k].get('caption_label',''),re.I)]
            if split=='train':candidates=[k for k in candidates if ann[k]['path'] not in valfamilies]
            candidates.sort(key=lambda k:hashlib.sha256(('20261003'+TASKS[task]+k).encode()).hexdigest())
            seen=set();count=0
            for key in candidates:
                if ann[key]['path'] in seen:continue
                try:
                    item=ds.load_keyid(key);batch=ds.collate_fn([item]);n=FRAMES[task];x=batch['x'][0];mask=batch['stats_mask'][0]
                    # Fixed first task window; pad short clips by holding their last pose, zeroing increments.
                    x=x[:n].clone();mask=mask[:n].clone()
                    if len(x)<n:
                        tail=x[-1:].repeat(n-len(x),1);tail[:,1:4]=0;x=torch.cat([x,tail]);mask=torch.cat([mask,mask[-1:].repeat(n-len(mask),1)])
                    with torch.no_grad():p=fk(x[None,:,:205]);q=quantity(p,task,HH/fk.height)[0]
                    if not torch.isfinite(q) or not torch.isfinite(x).all():continue
                    path=OUT/'cache'/f'{split}_{TASKS[task]}_{key}.pt'
                    tx={k:v.cpu() if torch.is_tensor(v) else v for k,v in batch['tx'].items()}
                    torch.save(dict(x=x,stats_mask=mask,tx=tx,quantity=q,task=task,key=key),path)
                    records.append(dict(split=split,task=TASKS[task],task_id=task,key=key,family=ann[key]['path'],caption=ann[key].get('caption_label'),quantity=float(q),path=str(path)))
                    seen.add(ann[key]['path']);count+=1
                    if len(parity)<3:
                        raw=x[:3,:205];bd=smplrifkefeats_to_smpldata(raw);exact=smpl(bd['poses'],bd['trans'],jointstype='smpljoints')[:,:22]
                        got=fk(raw[None],canonical=False)[0,:,:22];parity.append(float((exact-got).abs().max()))
                    if count>=limit:break
                except (FileNotFoundError,KeyError,ValueError) as e:print('skip',key,repr(e),flush=True)
            print(split,TASKS[task],count,flush=True)
        del ds
    save('data_manifest.json',records)
    audit={}
    for task in TASKS:
        audit[task]={}
        for split in ['train','val']:
            q=[r['quantity'] for r in records if r['task']==task and r['split']==split]
            audit[task][split]=dict(n=len(q),min=min(q) if q else None,max=max(q) if q else None)
    assert max(parity)<1e-4,parity  # <0.1 mm accommodates upstream matrix/axis-angle roundtrip in float32.
    save('data_audit.json',dict(tasks=audit,fk_parity_max_m=max(parity),human_shoulder_ankle_height_m=fk.height,display_height_m=HH,note='Automatically selected semantic captions; no manual certification. Family-disjoint adapter train/val. Frozen base retains its original pretraining exposure.'))
if __name__=='__main__':main()
