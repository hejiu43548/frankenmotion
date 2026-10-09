"""Motion-grounded admission of walking turns; never rewrites training data."""
import argparse, collections, hashlib, json, re, sys, time
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[2]
WALK=re.compile(r'\b(?:walk\w*|stroll\w*)\b',re.I)
BAD=re.compile(r'\b(?:run|runs|running|jog\w*|jump\w*|hop\w*|crawl\w*|dance\w*|dancing|climb\w*|stair\w*|kick\w*|sit\w*|kneel\w*)\b',re.I)
RULES=dict(version='walking_turn_v2',fps=20,smooth_frames=9,event_yaw_rate_rad_s=.12,event_gap_frames=7,min_event_angle_rad=.35,event_context_frames=12,min_crop_frames=40,max_crop_frames=120,min_walking_annotation_fraction=.85,min_path_m=.5,min_excursion_m=.3,min_speed_m_s=.2,max_speed_m_s=1.8,moving_speed_m_s=.15,min_moving_during_turn_fraction=.7,min_net_body_turn_rad=.35,max_net_body_turn_rad=3.5,min_turn_direction_consistency=.7,min_forward_alignment_fraction=.7,min_path_heading_turn_rad=.25,max_body_path_turn_difference_rad=.6,min_ankle_relative_excursion_m=.08,max_root_height_range_m=.2,min_event_path_m=.3,min_event_excursion_m=.2,min_yaw_weighted_moving_fraction=.8)
def dump(p,x):p.write_text(json.dumps(x,ensure_ascii=False,indent=2)+'\n')
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def smooth(x,n=9):
    if x.ndim==1:return np.convolve(np.pad(x,(n//2,n//2),mode='edge'),np.ones(n)/n,mode='valid')
    return np.stack([smooth(x[:,i],n) for i in range(x.shape[1])],-1)
def signals(pos):
    xy=pos[:,0,:2]; side=pos[:,1,:2]-pos[:,2,:2]
    heading=smooth(np.unwrap(np.arctan2(side[:,1],side[:,0])-np.pi/2))
    velocity=np.gradient(smooth(xy),axis=0)*20
    speed=np.linalg.norm(velocity,axis=-1)
    return xy,heading,velocity,speed,np.gradient(heading)*20

def events(rate):
    found=[]
    for sign in [-1,1]:
        idx=np.where(rate*sign>RULES['event_yaw_rate_rad_s'])[0]
        if not len(idx):continue
        start=last=int(idx[0])
        for i in idx[1:]:
            i=int(i)
            if i-last>RULES['event_gap_frames'] or np.any(rate[last:i]*sign<-.12):
                found.append((start,last+1,sign));start=i
            last=i
        found.append((start,last+1,sign))
    return sorted(found)

def evaluate(pos,walking,bad,ev):
    xy,h,v,speed,rate=signals(pos);a,b,sgn=ev
    L,R=max(0,a-12),min(len(pos),b+12)
    if R-L<40:
        need=40-(R-L);L=max(0,L-need//2);R=min(len(pos),L+40);L=max(0,R-40)
    P=pos[L:R];xy=xy[L:R];hh=h[L:R];vv=v[L:R];ss=speed[L:R];rr=rate[L:R];ea,eb=a-L,b-L
    if len(P)<3:return None
    event_xy=xy[max(0,ea):min(len(xy),eb)]
    event_path=float(np.linalg.norm(np.diff(event_xy,axis=0),axis=-1).sum());event_exc=float(np.linalg.norm(event_xy-event_xy[0],axis=-1).max())
    delta=float(hh[-1]-hh[0]);travel=float(np.linalg.norm(np.diff(xy,axis=0),axis=-1).sum());exc=float(np.linalg.norm(xy-xy[0],axis=-1).max())
    active=np.abs(rr)>.12;variation=float(np.abs(np.diff(hh)).sum())
    # Robust direction over 0.4 s before and after the detected event (within crop).
    v0=vv[max(0,ea-8):max(1,ea)].mean(0);v1=vv[min(len(vv)-1,eb):min(len(vv),eb+8)].mean(0)
    path_delta=float(np.arctan2(v0[0]*v1[1]-v0[1]*v1[0],np.dot(v0,v1)))
    # Resolve the U-turn +/-pi branch by body turn direction, never by arbitrary endpoint wrapping.
    if abs(path_delta)>2.8 and path_delta*delta<0:path_delta+=np.sign(delta)*2*np.pi
    forward=np.stack([np.cos(hh),np.sin(hh)],-1);cos=(forward*vv).sum(-1)/np.maximum(ss,1e-8);moving=ss>.15
    ankle=P[:,[7,8],:2]-P[:,None,0,:2];swing=np.linalg.norm(np.ptp(ankle,axis=0),axis=-1)
    m=dict(event_path_m=event_path,event_excursion_m=event_exc,yaw_weighted_moving_fraction=float(np.abs(rr[ss>.15]).sum()/max(np.abs(rr).sum(),1e-8)),body_turn_rad=-delta,path_turn_rad=-path_delta,event_turn_rad=float(-(h[b-1]-h[a])),path_m=travel,excursion_m=exc,net_displacement_m=float(np.linalg.norm(xy[-1]-xy[0])),mean_speed_m_s=travel/((len(P)-1)/20),moving_during_turn_fraction=float((ss[active]>.15).mean()) if active.any() else 0.,direction_consistency=abs(delta)/max(variation,1e-8),forward_alignment_fraction=float((cos[moving]>.5).mean()) if moving.any() else 0.,entry_speed_m_s=float(np.linalg.norm(v0)),exit_speed_m_s=float(np.linalg.norm(v1)),walking_annotation_fraction=float(walking[L:R].mean()),conflicting_action_fraction=float(bad[L:R].mean()),ankle_relative_excursion_m=swing.tolist(),root_height_range_m=float(np.ptp(P[:,0,2])))
    checks={'insufficient_event_translation':event_path>=.3 and event_exc>=.2,'rotation_concentrated_while_stationary':m['yaw_weighted_moving_fraction']>=.8,'crop_outside_2_6s':40<=len(P)<=120,'insufficient_walk_annotation':m['walking_annotation_fraction']>=.85,'conflicting_action':m['conflicting_action_fraction']==0,'insufficient_translation':travel>=.5 and exc>=.3,'walking_speed_outside_range':.2<=m['mean_speed_m_s']<=1.8,'stationary_during_turn':m['moving_during_turn_fraction']>=.7,'turn_angle_outside_range':.35<=abs(delta)<=3.5,'reversing_or_oscillating_turn':m['direction_consistency']>=.7,'not_forward_walking':m['forward_alignment_fraction']>=.7,'no_moving_entry_exit':min(m['entry_speed_m_s'],m['exit_speed_m_s'])>=.15,'body_turn_without_path_turn':abs(path_delta)>=.25 and delta*path_delta>0 and abs(delta-path_delta)<=.6,'weak_leg_motion':min(swing)>=.08,'large_vertical_motion':m['root_height_range_m']<=.2}
    return L,R,m,[k for k,v in checks.items() if not v]

def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args();a.out.mkdir(parents=True,exist_ok=False)
    sys.path.insert(0,str(ROOT/'work/unified_direct'));from common import Skeleton
    sys.path.insert(0,str(ROOT/'prepare/amasstools'));from fix_fps import interpolate_fps_poses,interpolate_fps_trans
    import torch
    torch.set_num_threads(2)
    annroot=ROOT/'datasets/annotations/frankenstein-dataset/annotations';ann=json.loads((annroot/'annotations.json').read_text())
    splitids={s:(annroot/'splits'/f'{s}.txt').read_text().split() for s in ['train','val','test']}
    fam={s:{ann[k]['path'] for k in ids} for s,ids in splitids.items()};assert not any(fam[x]&fam[y] for x,y in [('train','val'),('train','test'),('val','test')])
    rawindex=json.loads((ROOT/'outputs_amass/stage2_manifests_20261008/annotated_source_raw_audit.json').read_text())
    sk=Skeleton(ROOT/'outputs_amass/transfer_charlie_20261008/snapshot/outputs_amass/franken_eleven_20261003/skeleton.npz')
    groups=collections.defaultdict(list);excluded=[];all_ann=[]
    for split,ids in splitids.items():
        for key in ids:
            r=ann[key];walking=WALK.search(r.get('caption_label','')) or any(x['bodypart'] in ['action','left_leg','right_leg'] and WALK.search(x['text']) for x in r['annotations'])
            if walking:groups[r['path']].append((split,key,r))
    admitted=[];review=[];source_records=[];starttime=time.time()
    for fi,(family,rrr) in enumerate(sorted(groups.items())):
        info=rawindex.get(family,{});cached=ROOT/'outputs_amass/unified_direct_20261008/data/motions'/(family+'.npy')
        try:
            assert 'humanact12' not in family,'excluded_humanact12'
            assert info.get('status')=='raw_header_valid','raw_source_unavailable_or_invalid'
            if cached.exists():
                raw=np.load(cached);pos=sk(torch.from_numpy(raw)[None],canonical=False)[0].numpy();mode='existing_verified_full_RiFKE';path=cached
            else:
                path=Path(info['raw_path'])
                with np.load(path,allow_pickle=False) as z:
                    poses=torch.tensor(z['poses'][:,:66],dtype=torch.float32);trans=torch.tensor(z['trans'],dtype=torch.float32)
                poses=interpolate_fps_poses(poses,info['fps'],20);trans=interpolate_fps_trans(trans,info['fps'],20)
                pos=sk.raw(poses,trans).numpy();mode='raw_AMASS_resampled_existing_slerp_FK'
            assert np.isfinite(pos).all(),'nonfinite_motion'
            source_records.append(dict(family=family,split=rrr[0][0],path=str(path),sha256=sha(path),mode=mode,frames=len(pos)))
        except Exception as e:
            for split,key,r in rrr:excluded.append(dict(split=split,key=key,family=family,reason=str(e)))
            continue
        for split,key,r in rrr:
            st=max(0,int(r['start']*20));end=min(len(pos),int(r['end']*20));p=pos[st:end]
            base=dict(split=split,key=key,family=family,caption=r.get('caption_label',''),raw_path=info['raw_path'],annotation_start_s=r['start'],annotation_end_s=r['end'])
            if len(p)<40:excluded.append(dict(base,reason='annotation_under_2s'));continue
            walking=np.zeros(len(p),bool);bad=np.zeros(len(p),bool)
            for x in r['annotations']:
                mask=(np.arange(st,end)/20>=x['start']) & (np.arange(st,end)/20<x['end'])
                # Part annotations use absolute source time, as the released text loader does.
                if x['bodypart'] in ['action','left_leg','right_leg']:
                    if WALK.search(x['text']) and x.get('confidence',0)>=3:walking[mask]=True
                    if BAD.search(x['text']):bad[mask]=True
            if not walking.any():review.append(dict(base,reasons=['missing_timed_walk_annotation']));continue
            _,h,_,_,rate=signals(p);evs=[]
            for l,u,sign in events(rate):
                if abs(h[u-1]-h[l])<.35:continue
                # Long curves are represented by bounded event chunks, not cropped at annotation start.
                for x in range(l,u,96):
                    y=min(u,x+96)
                    if y-x>=6 and abs(h[y-1]-h[x])>=.35:evs.append((x,y,sign))
            if not evs:excluded.append(dict(base,reason='no_sustained_turn_event'));continue
            for ev in evs:
                result=evaluate(p,walking,bad,ev)
                if result is None:continue
                l,u,metrics,reasons=result
                if re.search(r'in[- ]place|on the spot|on (?:one|a single) foot',base['caption'],re.I):reasons.append('explicit_stationary_or_single_foot_pivot_text')
                row=dict(base,task='turn',event_start_frame_20fps=st+ev[0],event_end_frame_20fps=st+ev[1],crop_start_frame_20fps=st+l,crop_end_frame_20fps=st+u,real_frames=u-l,pad_frames=0,direction='right' if metrics['body_turn_rad']>0 else 'left',metrics=metrics,admission_reasons=reasons,manual_verified=False,ready_for_training=False)
                row['id']=hashlib.sha256(f'{family}:{st+l}:{st+u}'.encode()).hexdigest()[:16]
                if reasons:review.append(row)
                else:admitted.append(row)
        if (fi+1)%200==0:print(json.dumps(dict(families=fi+1,total=len(groups),admitted=len(admitted),review=len(review),seconds=time.time()-starttime)),flush=True)
    # Merge duplicate/near-duplicate event windows from overlapping source annotations.
    final=[];duplicates=[]
    for row in sorted(admitted,key=lambda r:(r['family'],-r['metrics']['walking_annotation_fraction'],-r['metrics']['direction_consistency'],r['key'])):
        duplicate=None
        for old in final:
            if old['family']!=row['family']:continue
            overlap=max(0,min(old['crop_end_frame_20fps'],row['crop_end_frame_20fps'])-max(old['crop_start_frame_20fps'],row['crop_start_frame_20fps']))
            if overlap/min(old['real_frames'],row['real_frames'])>=.5:duplicate=old;break
        if duplicate:
            duplicates.append(dict(row,selected_id=duplicate['id']));duplicate.setdefault('alternative_annotations',[]).append(row['key'])
        else:final.append(row)
    for split in splitids:dump(a.out/f'turn_{split}.json',[r for r in final if r['split']==split])
    dump(a.out/'admitted_all.json',final);dump(a.out/'review.json',review);dump(a.out/'excluded_annotations.json',excluded);dump(a.out/'duplicate_events.json',duplicates);dump(a.out/'sources.json',source_records)
    summary=dict(annotation_universe=len(ann),walking_annotations=sum(map(len,groups.values())),walking_source_families=len(groups),source_loaded=len(source_records),accepted_before_dedup=len(admitted),accepted_events=len(final),by_split={s:dict(events=sum(r['split']==s for r in final),families=len({r['family'] for r in final if r['split']==s}),directions=dict(collections.Counter(r['direction'] for r in final if r['split']==s))) for s in splitids},review_events=len(review),review_reasons=dict(collections.Counter(x for r in review for x in r.get('reasons',r.get('admission_reasons',[])))),excluded_reasons=dict(collections.Counter(r['reason'] for r in excluded)),deduplicated_events=len(duplicates),seconds=time.time()-starttime)
    dump(a.out/'summary.json',summary);dump(a.out/'rules.json',RULES);dump(a.out/'provenance.json',dict(script_sha256=sha(__file__),annotations_sha256=sha(annroot/'annotations.json'),splits_sha256={s:sha(annroot/'splits'/f'{s}.txt') for s in splitids},skeleton_sha256=sha(ROOT/'outputs_amass/transfer_charlie_20261008/snapshot/outputs_amass/franken_eleven_20261003/skeleton.npz'),scope='All annotated walking sources; unannotated AMASS not semantically assigned. Only selects manifests; no cached motion/text regeneration or model training. Rule-based admitted candidates require visual QA. Existing datasets unchanged.'))
    print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
