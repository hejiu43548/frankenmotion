import json,subprocess
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/universal_tracker_20261006';W=R/'work/universal_tracker_20261006';py=R/'work/mjlab_stable_env/bin/python';base=D/'general_evaluation/frozen_demo_service_trimmed';f=json.loads((D/'frozen_unified/protocol.json').read_text());assert json.loads((base/'protocol.json').read_text())['checkpoint_sha256']==f['checkpoint_sha256'];out=D/'visuals/frozen_service_trimmed';out.mkdir(exist_ok=False)
for row in json.loads((base/'results.json').read_text()):
 run=Path(row['run']);name=run.name
 if not row['physical_complete']:continue
 for label,script,args in [('presentation','render_navigation_presentation.py',['--run',run,'--output',out/(name+'.mp4')]),('reference','render_general.py',['--run',run,'--output',out/(name+'_comparison'),'--frames-only','--follow']),('walk','render_temporal_strip.py',['--run',run,'--output',out/(name+'_walk.png'),'--start',1,'--end',4,'--count',20])]:
  with (out/(name+'_'+label+'.log')).open('w') as log:subprocess.run([str(py),str(W/script),*map(str,args)],cwd=R,stdout=log,stderr=subprocess.STDOUT,check=True)
 print(name,'rendered; visual review pending',flush=True)
