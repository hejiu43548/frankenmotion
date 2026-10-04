"""Matched short/long preview continuation, with a global development-selected parent."""
import datetime,hashlib,json,subprocess,time
from pathlib import Path
R=Path('/home/pku/frankenmotion');U=R/'outputs_amass/franken_unified_20261004';P=R/'work/mjlab_stable_env/bin/python';C=R/'work/g1_sim_env/bin/python'
read=lambda p:json.loads(p.read_text())
names=[f'joint_v4_tracking_step{i}_validation' for i in [1000,2000,3000,4000]]+['joint_v4_tracking_validation']
assert not (U/'paired_preview_protocol.json').exists()
while not all((U/'evaluation'/n/'audit.json').exists() for n in names):
    if not (U/'training/joint_v4_tracking/complete.json').exists():
        assert Path('/proc/393567/cmdline').exists(), 'V4 ended without training completion'
    time.sleep(20)
candidates=[]
for n in names:
    f=U/'evaluation'/n;a=read(f/'audit.json');p=read(f/'protocol.json')
    assert a['aggregate']['requests']==110 and a['raw_max_error']<1e-8 and a['overflow_warnings']==0 and not p['task_routing']
    candidates.append(dict(name=n,score=a['aggregate']['macro_semantic_E_all'],checkpoint=p['checkpoint'],sha256=p['sha256']))
best=min(candidates,key=lambda x:(x['score'],x['name']));initial=Path(best['checkpoint']);assert hashlib.sha256(initial.read_bytes()).hexdigest()==best['sha256']
long_initial=U/'initial/long_preview_v5_parent.pt'
subprocess.run([str(P),str(R/'work/unified_expand_checkpoint_20261004.py'),'--source',str(initial),'--destination',str(long_initial),'--long-preview'],cwd=R,check=True)
common=['--dataset','training_corpus_augmented','--preview','--calibrate-preview','--reset-training-rng','--task-balanced-slots','--steps','5000','--envs','520','--seed','5104','--body-pos-std','.1','--body-pos-weight','2','--joint-weight','2','--joint-worst-count','8','--action-rate-weight','-.02','--entropy-coef','.003','--episode-seconds','30','--learning-rate','5e-5','--root-ori-weight','2','--root-wide-weight','1']
protocol=dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),parent_candidates=candidates,parent=best,common_arguments=common,contrast='Shared short .1/.2/.4s versus shared long .1/.2/.4/.7/1s preview. Both use identical stronger global root rewards, fresh optimizer, matched reset seed and training corpus. No task routing. One training seed; not multi-seed statistical evidence.',runs={})
(U/'paired_preview_protocol.json').write_text(json.dumps(protocol,indent=2))
jobs=[]
for name,ck,extra in [('joint_v5_root_short',initial,[]),('joint_v5_root_long',long_initial,['--long-preview'])]:
    log=(R/f'work/unified_{name}.log').open('w');args=[str(P),str(R/'work/unified_train_20261004.py'),'--name',name,'--initial',str(ck)]+common+extra
    proc=subprocess.Popen(args,cwd=R,stdout=log,stderr=subprocess.STDOUT)
    watcherlog=(R/f'work/unified_watch_{name}.log').open('w');watch=subprocess.Popen([str(P),str(R/'work/unified_watch_checkpoints_20261004.py'),'--train-name',name,'--pid',str(proc.pid),'--preview','--indices','1000','3000'],cwd=R,stdout=watcherlog,stderr=subprocess.STDOUT)
    protocol['runs'][name]=dict(pid=proc.pid,watcher_pid=watch.pid,arguments=args);(U/'paired_preview_protocol.json').write_text(json.dumps(protocol,indent=2));jobs.append((name,proc,watch))
    print('Started',name,proc.pid,flush=True)
    time.sleep(10)
for name,proc,watch in jobs:
    assert proc.wait()==0,f'{name} failed; inspect log'
    checkpoint=U/'training'/name/'model_4999.pt';evalname=name+'_validation'
    with (R/f'work/unified_{name}_eval.log').open('w') as log:
        subprocess.run([str(P),str(R/'work/unified_evaluate_20261004.py'),'--checkpoint',str(checkpoint),'--preview','--name',evalname],cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
        subprocess.run([str(C),str(R/'work/unified_assess_20261004.py'),'--name',evalname],cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
    assert watch.wait()==0
    print('Completed and audited',name,flush=True)
(U/'paired_preview_complete.json').write_text(json.dumps(dict(names=[x[0] for x in jobs])))
