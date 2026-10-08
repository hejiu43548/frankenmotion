"""Seen-training-reference fidelity diagnostic, not validation or final testing."""
import json,time,hashlib,subprocess
from pathlib import Path
import numpy as np
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/universal_tracker_20261006';W=R/'work/universal_tracker_20261006';py=R/'work/mjlab_stable_env/bin/python';manifest=D/'natural_training_diagnostic_manifest.json';assert not manifest.exists();rows=json.loads((D/'natural_extended_corrected_motion/manifest.json').read_text());rng=np.random.default_rng(610611);selected=[]
for task in sorted({r['task'] for r in rows}):
 train=[r for r in rows if r['task']==task and r['split']=='train'];indices=rng.permutation(len(train))[:5];selected.extend(train[int(i)] for i in indices)
assert len({r['source'] for r in selected})==len(selected);assert all(r['split']=='train' for r in selected);manifest.write_text(json.dumps(selected,indent=2));(D/'training_diagnostic_protocol.json').write_text(json.dumps(dict(scope=__doc__,seed=610611,per_category_max=5,requests=len(selected),entry='RSI: original reference pose and velocity, artificial standing prefix removed',purpose='Distinguish poor tracking of seen references from failure only on new sources. Never used as a final success claim or checkpoint selection score.'),indent=2))
for name,ck in [('v4_3000',D/'training/joint_v4_long_physical/model_3000.pt'),('v5_1000',D/'training/joint_v5_expanded/model_1000.pt'),('v5_3000',D/'training/joint_v5_expanded/model_3000.pt'),('v5_final',D/'training/joint_v5_expanded/model_5999.pt')]:
 while not ck.with_suffix('.pt.ready.json').exists():
  if time.time()>1791247800:raise RuntimeError('Diagnostic cutoff00:50UTC')
  time.sleep(10)
 assert hashlib.sha256(ck.read_bytes()).hexdigest()==json.loads(ck.with_suffix('.pt.ready.json').read_text())['checkpoint_sha256'];tag='training_reference_'+name
 with (D/(tag+'.log')).open('w') as log:
  subprocess.run([str(py),str(W/'evaluate_cpu_general.py'),'--checkpoint',str(ck),'--manifest',str(manifest),'--name',tag,'--workers','4','--entry','rsi'],cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
  subprocess.run([str(py),str(W/'assess_general_fidelity.py'),'--run',str(D/'general_evaluation'/tag)],cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
 print(tag,'complete',flush=True)
