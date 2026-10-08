import sys,json,subprocess
from pathlib import Path
R=Path('/home/pku/frankenmotion');W=R/'work/jump_tracker_20261007';D=R/'outputs_amass/jump_tracker_20261007';py=R/'work/mjlab_stable_env/bin/python';a=json.loads((R/'outputs_amass/universal_tracker_20261006/training/joint_v5_expanded/protocol.json').read_text())['arguments'];name=sys.argv[1];phase=float(sys.argv[2]);steps=int(sys.argv[3]);a.update(name=name,initial=str(D/'backup/policy.pt'),steps=steps,seed=7107,learning_rate=5e-5,retention_weight=.25,phase_weight=phase,jump_fraction=.4,start_probability=.7,termination_tolerance=.6)
if len(sys.argv)>4:a.update(json.loads(Path(sys.argv[4]).read_text()))
command=[py,W/'train_jump.py']
for k,v in a.items():
 if isinstance(v,bool):
  if v:command.append('--'+k.replace('_','-'))
 elif v is not None:command += ['--'+k.replace('_','-'),str(v)]
raise SystemExit(subprocess.call([str(py),str(W/'run_budgeted.py'),'--label',name,*map(str,command)]))
