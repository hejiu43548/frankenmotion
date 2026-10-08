import sys,time,subprocess,json
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/unified_generator_20261007';W=R/'work/unified_generator_20261007';py=R/'.conda/bin/python';auditpy=R/'work/mjlab_stable_env/bin/python'
def run(name,checkpoint=None,teachers=False):
 cmd=[str(py),str(W/'generate_eval.py'),'--name',name]
 cmd+=['--teachers'] if teachers else ['--checkpoint',str(checkpoint)]
 if not (D/'generation'/name/'complete.json').exists():
  with (D/(name+'_generation.log')).open('w') as f:subprocess.run(cmd,cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
 subprocess.run([str(auditpy),str(W/'assess.py'),'--name',name],cwd=R,check=True)
run('teacher_development',teachers=True)
run('physical_development',D/'backup/physical.pt')
for step in [2200,6600,11000]:
 ck=D/'training/shared_v1'/f'model_{step:05d}.pt'
 while not ck.with_suffix('.ready.json').exists():
  status=D/'shared_v1_process.json'
  if status.exists() and json.loads(status.read_text())['returncode']!=0:raise RuntimeError('Training failed')
  time.sleep(5)
 run(f'shared_v1_{step}_development',ck)
