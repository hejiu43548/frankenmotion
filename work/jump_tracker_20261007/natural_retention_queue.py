import sys,json,subprocess
from pathlib import Path
import numpy as np
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/jump_tracker_20261007';W=R/'work/jump_tracker_20261007';py=sys.executable
subprocess.run([py,str(W/'prepare_natural_retention.py')],check=True)
subprocess.run([py,str(W/'collect_pulse_teacher.py'),'--checkpoint',str(D/'backup/policy.pt'),'--manifest',str(D/'natural_retention/manifest.json'),'--name','natural_retention','--workers','4'],check=True)
rows=json.load(open(D/'general_evaluation/natural_retention/results.json'));obs=[];records=[]
for r in rows:
 if 'error' in r:continue
 z=np.load(Path(r['run'])/'pulse.npz');x=z['observations'];x=x[np.isfinite(x).all(1)];obs.append(x);records.append(dict(task=r['task'],source=r['source'],run=r['run'],frames=len(x),physical_complete=r['physical_complete']))
np.savez_compressed(D/'natural_retention/observations.npz',observations=np.concatenate(obs).astype(np.float32));(D/'natural_retention/records.json').write_text(json.dumps(records,indent=2));print('RETENTION',len(records),sum(r['frames'] for r in records),flush=True)
subprocess.run([py,str(W/'train_residual_actor.py'),'--teacher',str(D/'general_evaluation/planned_teacher_train/audited_results.json'),'--name','reference_residual_v4_broad','--steps','15000','--reference-only','--hidden-width','256','--learning-rate','.0002','--cosine','--seed','710711','--extra-retention',str(D/'natural_retention/observations.npz')],check=True)
for split,manifest in [('dev',D/'dev_jump_manifest.json'),('extended_dev',D/'extended_dev_jump_manifest.json'),('regression',R/'outputs_amass/universal_tracker_20261006/native_eleven_manifest.json'),('natural',R/'outputs_amass/universal_tracker_20261006/natural_test_motion/manifest.json')]:
 actor=D/'residual_training/reference_residual_v4_broad/actor_14999.pt';name='residual_v4_'+split
 subprocess.run([py,str(W/'evaluate_actor.py'),'--checkpoint',str(actor),'--actor',str(actor),'--manifest',str(manifest),'--name',name,'--workers','4'],check=True)
 if split=='natural':subprocess.run([py,str(R/'work/universal_tracker_20261006/assess_general_fidelity.py'),'--run',str(D/'general_evaluation'/name)],check=True)
 else:subprocess.run([py,str(W/'assess.py'),'--name',name],check=True)
