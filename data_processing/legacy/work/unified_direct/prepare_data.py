"""Rebuild direct-supervision data from raw AMASS; never load old adapters."""
import argparse
from concurrent.futures import ProcessPoolExecutor
import os
import time
from common import *


def convert(job):
    family,raw_path,dest,skeleton = job
    try:
        torch.set_num_threads(1)
        sys.path.insert(0,str(ROOT/'prepare/amasstools'))
        from fix_fps import interpolate_fps_poses,interpolate_fps_trans
        from src.tools.smplrifke_feats import smpldata_to_smplrifkefeats,smplrifkefeats_to_smpldata
        path=Path(dest)
        if path.exists(): return dict(family=family,status='cached')
        with np.load(raw_path,allow_pickle=False) as z:
            poses=torch.tensor(z['poses'][:,:66],dtype=torch.float32)
            trans=torch.tensor(z['trans'],dtype=torch.float32)
            fps=float(z['mocap_framerate'] if 'mocap_framerate' in z else z['mocap_frame_rate'])
        assert torch.isfinite(poses).all() and torch.isfinite(trans).all(),'nonfinite_source'
        poses=interpolate_fps_poses(poses,fps,20).double()
        trans=interpolate_fps_trans(trans,fps,20).double()
        assert len(poses)>=3,'source_too_short'
        sk=Skeleton(skeleton).double();joints=sk.raw(poses,trans)
        feat=smpldata_to_smplrifkefeats(dict(poses=poses,trans=trans,joints=joints)).float()
        assert torch.isfinite(feat).all(),'nonfinite_features'
        decoded=smplrifkefeats_to_smpldata(feat)['joints']
        fk=sk.float()(feat[None],canonical=False)[0]
        error=float((decoded-fk).abs().max())
        assert error<2e-4,('FK_roundtrip',error)
        path.parent.mkdir(parents=True,exist_ok=True)
        temp=path.with_suffix('.tmp.npy');np.save(temp,feat.numpy());temp.replace(path)
        return dict(family=family,status='ok',frames=len(feat),fk_error=error)
    except Exception as e:return dict(family=family,status='rejected',reason=repr(e))


def main():
    p=argparse.ArgumentParser();p.add_argument('--out',type=Path,required=True)
    p.add_argument('--manifests',type=Path,required=True);p.add_argument('--skeleton',type=Path,required=True)
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--workers',type=int,default=4)
    a=p.parse_args();a.out.mkdir(parents=True,exist_ok=True);torch.set_num_threads(1);np.random.seed(1234)
    rows=sum([json.loads((a.manifests/f'full_candidates_{s}.json').read_text()) for s in ['train','val']],[])
    sources={r['family']:r['raw_path'] for r in rows};start=time.time()
    jobs=[(f,raw,str(a.out/'motions'/(f+'.npy')),str(a.skeleton)) for f,raw in sources.items()]
    audit=[]
    with ProcessPoolExecutor(a.workers) as pool:
        for i,result in enumerate(pool.map(convert,jobs,chunksize=4)):
            audit.append(result)
            if (i+1)%100==0:
                status=dict(state='motion_conversion',done=i+1,total=len(jobs),seconds=time.time()-start)
                save_json(a.out/'prepare_status.json',status);print(status,flush=True)
    save_json(a.out/'source_audit.json',audit)
    bad={x['family']:x['reason'] for x in audit if x['status']=='rejected'}
    from src.data.text_part import TextEmbeddings
    config=json.loads((a.checkpoint/'config.json').read_text())['data']['text_encoder'].copy()
    config.pop('_target_');config['rand_mask']=False
    text=TextEmbeddings(**config)
    np.savez(a.out/'pca.npz',mean=text.pca_components.mean_,components=text.pca_components.components_)
    ann_path=ROOT/'datasets/annotations/frankenstein-dataset/annotations/annotations.json'
    annotations=json.loads(ann_path.read_text());sk=Skeleton(a.skeleton)
    reduced={};accepted=[];rejected=[];parity=[]
    def embed(label):
        assert label in text.embeddings_index,('missing_text',label)
        e=text.get_embedding(label)['x'];assert e.shape==(1,512),e.shape
        if label not in reduced:reduced[label]=text.pca_components.transform(e.numpy())[0]
        return e.numpy()[0],reduced[label]
    for i,r in enumerate(rows):
        try:
            assert r['family'] not in bad,bad.get(r['family'])
            motion=np.load(a.out/'motions'/(r['family']+'.npy'))
            n=r['target_frames'];s=r['crop_start_frame_20fps'];end=min(r['annotation_end_frame_20fps'],s+n,len(motion))
            motion=motion[s:end].copy();real=len(motion);assert real>=3
            if real<n:
                motion=np.concatenate([motion,np.repeat(motion[-1:],n-real,axis=0)])
                motion[real-1:,1:4]=0
            ann=annotations[r['key']]
            # Match released loader's full annotation timing before applying the task crop.
            runs=text._format_annotations(ann['annotations'],ann['start'],ann['end'],20)
            duration=int(20*(ann['end']-ann['start']));local=np.zeros((duration,408),np.float32);mask=np.zeros_like(local,dtype=bool)
            for part_idx,part in enumerate(text.body_part_order):
                assert part in runs,('missing_part',part)
                at=0
                for label,count in runs[part]:
                    e,red=embed(label);stop=min(at+count,duration)
                    if label!='unknown':local[at:stop,part_idx*51:(part_idx+1)*51]=red;mask[at:stop,part_idx*51:(part_idx+1)*51]=True
                    at=stop
            global_label=runs['sequence_caption'][-1][0];tx,_=embed(global_label)
            if len(parity)<3:
                ref,_=text.load_from_annotation(ann['annotations'],start=ann['start'],end=ann['end'])
                err=float(np.max(np.abs(ref['local']['x'].numpy()-local)))
                assert err<1e-4 and np.array_equal(ref['local']['mask'].numpy(),mask)
                assert np.allclose(ref['x'].numpy()[0],tx);parity.append(err)
            if len(local)<n:
                local=np.concatenate([local,np.repeat(local[-1:],n-len(local),0)])
                mask=np.concatenate([mask,np.repeat(mask[-1:],n-len(mask),0)])
            local=local[:n];mask=mask[:n]
            with torch.no_grad():q=float(quantity(sk(torch.from_numpy(motion)[None]),r['task_id'],HUMAN_EQUIVALENT_HEIGHT/sk.height)[0])
            assert np.isfinite(q) and np.isfinite(local).all() and np.isfinite(tx).all()
            cache=a.out/'cache'/f"{r['split']}_{r['task']}_{r['key']}.npz";cache.parent.mkdir(exist_ok=True)
            np.savez(cache,motion=motion,local=local,local_mask=mask,tx=tx,quantity=np.float32(q),task=np.int64(r['task_id']))
            item=dict(r,cache=str(cache),quantity=q,real_frames=real,pad_frames=n-real,ready_for_training=True,
                      quantity_status='measured_on_actual_crop',pending_checks=[],
                      semantic_event_verification='not_manually_verified; caption matching only; target is measured crop quantity')
            accepted.append(item)
        except Exception as e:rejected.append(dict(r,reason=repr(e)))
        if (i+1)%200==0:
            status=dict(state='text_and_labels',done=i+1,total=len(rows),accepted=len(accepted),rejected=len(rejected),seconds=time.time()-start)
            print(status,flush=True);save_json(a.out/'prepare_status.json',status)
    for split in ['train','val']:save_json(a.out/f'{split}.json',[r for r in accepted if r['split']==split])
    save_json(a.out/'rejected.json',rejected)
    summary=dict(state='complete',accepted=len(accepted),rejected=len(rejected),seconds=time.time()-start,
                 text_parity_max_errors=parity,fk_max_error=max([x.get('fk_error',0) for x in audit]),
                 pca_sha256=sha(a.out/'pca.npz'),skeleton_sha256=sha(a.skeleton),annotation_sha256=sha(ann_path),
                 source_manifests={s:sha(a.manifests/f'full_candidates_{s}.json') for s in ['train','val']},
                 counts={s:{t:sum(r['split']==s and r['task']==t for r in accepted) for t in TASKS} for s in ['train','val']},
                 limitations=['Caption selection does not verify event presence in fixed crop. Labels are measured on that crop.',
                              'PCA refitted with seed 1234 and sklearn 1.3.1, as loader does; original training PCA not released.'])
    save_json(a.out/'prepare_status.json',summary);print(summary,flush=True)


if __name__=='__main__':main()
