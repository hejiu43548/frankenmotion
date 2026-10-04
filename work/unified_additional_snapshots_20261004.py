"""Complete the shared snapshot grid before the final cohort is created."""
import datetime,json,subprocess
from pathlib import Path
R=Path('/home/pku/frankenmotion');U=R/'outputs_amass/franken_unified_20261004';P=R/'work/mjlab_stable_env/bin/python'
assert not (U/'frozen_unified').exists() and not (U/'final_test').exists()
out=U/'additional_snapshot_protocol.json';assert not out.exists()
protocol=dict(created_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),reason='Development learning was non-monotonic: V4 step4000 outperformed its final snapshot. Measured source-generation/retarget runtime leaves time to evaluate both missing 2000/4000 checkpoints symmetrically. This extends the exploratory development search before any final data are created; not a retrospective claim of initial preregistration.',indices=[2000,4000],selection='One global all-11-task E_all score; no per-task winners',runs={})
out.write_text(json.dumps(protocol,indent=2));jobs=[]
for name,pid in [('joint_v5_root_short',399920),('joint_v5_root_long',400029)]:
 log=(R/f'work/unified_extra_{name}.log').open('w');proc=subprocess.Popen([str(P),str(R/'work/unified_watch_checkpoints_20261004.py'),'--train-name',name,'--pid',str(pid),'--preview','--indices','2000','4000'],cwd=R,stdout=log,stderr=subprocess.STDOUT)
 protocol['runs'][name]=dict(training_pid=pid,watcher_pid=proc.pid);out.write_text(json.dumps(protocol,indent=2));jobs.append(proc);print(name,proc.pid,flush=True)
for proc in jobs:assert proc.wait()==0
(U/'additional_snapshot_complete.json').write_text(json.dumps(dict(indices=[2000,4000],names=list(protocol['runs']))));print('Additional snapshot grid audited',flush=True)
