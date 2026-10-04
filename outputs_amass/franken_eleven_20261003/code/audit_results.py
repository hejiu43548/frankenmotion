import os,json,hashlib,csv
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
import transfer as tr
OUT=tr.OUT
TASKS=['raise_hand','reach','strike','wave','turn','sidestep','back_walk','kick','jump','lean','walk']
RANGES=[(.35,.75),(.25,.55),(1.5,2.5),(.08,.22),(.45,1.5),(.4,1.2),(.35,.9),(.25,.7),(.25,.55),(.4,.9),(.5,1.1)]
TOLS=[.02,.02,.1,.015,.08,.06,.05,.05,.04,.06,.05]
def canonical(p):
    side=p[0,1]-p[0,2];angle=np.arctan2(side[1],side[0])-np.pi/2
    return Rotation.from_euler('z',-angle).apply(p.reshape(-1,3)).reshape(p.shape)
def native_metrics(row):
    z=np.load(row['path']);p=canonical(z['joints_zup_m']);h=float(z['human_height']);r=tr.measure(p,row['task'],h)
    torso=(p[:,16]+p[:,17])/2-p[:,0];tilt=np.rad2deg(np.arctan2(np.linalg.norm(torso[:,:2],axis=-1),torso[:,2]));vel=np.linalg.norm(np.diff(p[:,0],axis=0),axis=-1)*20
    r.update(boundary_torso_over45=bool(np.any(np.r_[tilt[:5],tilt[-5:]]>45)),root_speed_peak_human_equivalent=float(vel.max()*tr.HH/h),cached_training_quantity=float(z['human_quantity']))
    return r
def stats(rows,key,span,tol):
    valid=[r for r in rows if r[key] is not None];errors=[abs(r[key]['quantity']-r['command']) for r in valid]
    q=[min(abs(r[key]['quantity']-r['command'])/span,1) if r[key] is not None else 1 for r in rows]
    slopes=[]
    for source in sorted(set(r['source'] for r in rows)):
        a=sorted([r for r in valid if r['source']==source],key=lambda r:r['command'])
        if len(a)==5:slopes.append(float(np.polyfit([r['command'] for r in a],[r[key]['quantity'] for r in a],1)[0]))
    return dict(planned=len(rows),measurable=len(valid),numeric_pass=sum(e<=tol for e in errors),event_pass=sum(bool(r[key]['event_pass']) for r in valid),joint_pass=sum(abs(r[key]['quantity']-r['command'])<=tol and bool(r[key]['event_pass']) for r in valid),mae_measurable=float(np.mean(errors)) if errors else None,E_all=float(np.mean(q)),complete_five_level_sources=len(slopes),median_source_slope=float(np.median(slopes)) if slopes else None)
def main():
    out=OUT/'delivery';out.mkdir(exist_ok=True)
    sims=json.loads((OUT/'simulation/task_adapter_results.json').read_text());old=json.loads((OUT/'generated/existing_adapter_manifest.json').read_text());old={(r['source'],r['command_index']):r for r in old}
    assert len(sims)==880 and len(old)==880
    model=tr.rt.load_model();rows=[];clock_errors=[]
    for r in sims:
        native=native_metrics(r);baseline=native_metrics(old[(r['source'],r['command_index'])]);row=dict(task=r['task'],source=r['source'],seed=r['seed'],command=r['command'],command_index=r['command_index'],old_human=baseline,human=native,g1=None,mode0=None,mode2=None,error=r.get('error'))
        if 'conversion' in r:
            folder=OUT/'simulation/task_adapter';label=Path(r['path']).stem;z=np.load(folder/(label+'_reference.npz'));n=len(z['q']);h=r['conversion']['robot_height_m']
            row['g1']=tr.measure(tr.get_positions(model,np.c_[z['root'],z['quat'],z['q']]),r['task'],h)
            for mode in ['0','2']:
                result=r['modes'].get(mode,{})
                if result.get('complete'):
                    raw=np.load(folder/(label+f'_mode{mode}.npz'));ps=tr.get_positions(model,raw['qpos']);time=raw['time_s'];desired=1+np.arange(n)*.05
                    assert time[-1]>=desired[-1]-1e-7
                    sampled=np.stack([np.interp(desired,time,x) for x in ps.reshape(len(ps),-1).T],1).reshape(n,24,3)
                    row['mode'+mode]=tr.measure(sampled,r['task'],h)
                    clock_errors.append(abs(row['mode'+mode]['quantity']-result['metrics']['quantity']))
        rows.append(row)
    summary={}
    for i,task in enumerate(TASKS):
        a=[r for r in rows if r['task']==task];assert len(a)==80 and len(set(r['source'] for r in a))==16
        summary[task]={key:stats(a,key,RANGES[i][1]-RANGES[i][0],TOLS[i]) for key in ['old_human','human','g1','mode0','mode2']}
        summary[task]['conversion_mae']=float(np.mean([abs(r['g1']['quantity']-r['human']['quantity']) for r in a if r['g1'] is not None]))
        summary[task]['human_boundary_tilt_flags']=sum(r['human']['boundary_torso_over45'] for r in a)
    flat=[]
    for r in rows:
        line={k:r[k] for k in ['task','source','seed','command','command_index']}
        for key in ['old_human','human','g1','mode0','mode2']:
            line[key]=None if r[key] is None else r[key]['quantity'];line[key+'_event']=None if r[key] is None else r[key]['event_pass']
        flat.append(line)
    with (out/'all_requests.csv').open('w') as f:
        writer=csv.DictWriter(f,fieldnames=flat[0]);writer.writeheader();writer.writerows(flat)
    audit=dict(planned=880,simulator_rollouts_planned=1760,conversion_errors=sum(r['error'] is not None for r in rows),raw_quantity_recalculation_max_error=max(clock_errors,default=0),strike_training_proxy_note='Training surrogate included one additional 50ms speed sample; ALL reported curves remeasure with the collaborator frozen 0.8–1.6s convention.',source_set='4 prompt templates x 4 held-out noise seeds per task; prompts shared with adapter finetuning, not unseen-text generalization.',controller='official default release, NOT partner SONIC v1.1; v1.1 models unavailable locally and official hosting was unreachable during this run.',retarget='joint-centre direction IK with analytic Jacobian, native root orientation and scaled translation; different adapter from partner GMR',height=dict(human_display=tr.HH,g1=tr.robot_height(model)),schema='3s/6s tasks at20Hz;1s entry,50Hz policy,500Hzphysics; missing physical completion -> missing quantity and E_all=1',summary=summary)
    (out/'summary.json').write_text(json.dumps(audit,indent=2,default=lambda x:x.item()));(out/'audited_results.json').write_text(json.dumps(rows,indent=2,default=lambda x:x.item()))
    print(json.dumps({t:{k:summary[t][k]['E_all'] for k in ['old_human','human','mode0','mode2']} for t in TASKS},indent=2))
if __name__=='__main__':main()
