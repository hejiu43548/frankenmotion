"""All pre-existing generated demo candidates, with one already frozen policy.
These are presentation candidates exposed during development, never a blind test.
"""
import sys,json,hashlib,subprocess
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/universal_tracker_20261006';W=R/'work/universal_tracker_20261006';py=R/'work/mjlab_stable_env/bin/python'
freeze=json.loads((D/'frozen_unified/protocol.json').read_text());checkpoint=Path(freeze['checkpoint']);assert hashlib.sha256(checkpoint.read_bytes()).hexdigest()==freeze['checkpoint_sha256'] and not freeze['task_routing'];logs=D/'frozen_demo_logs';logs.mkdir(exist_ok=False)
for name,dataset,anchors in [('point','generated_point_corrected_motion',False),('navigation','navigation_motion',True),('official_text','generated_official_motion',False),('service','service_motion',True)]:
 manifest=D/dataset/'manifest.json';assert manifest.exists();tag='frozen_demo_'+name
 command=[str(py),str(W/'evaluate_cpu_general.py'),'--checkpoint',str(checkpoint),'--manifest',str(manifest),'--name',tag,'--workers','4']+(['--anchors'] if anchors else [])
 with (logs/(name+'.log')).open('w') as f:subprocess.run(command,cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
 run=D/'general_evaluation'/tag
 subprocess.run([str(py),str(W/'assess_general_fidelity.py'),'--run',str(run)],cwd=R,check=True)
 if anchors:subprocess.run([str(py),str(W/'assess_navigation.py'),'--run',str(run)],cwd=R,check=True)
 if name=='official_text':subprocess.run([str(py),str(W/'audit_clap_geometry.py'),'--run',str(run)],cwd=R,check=True)
 (run/'presentation_scope.json').write_text(json.dumps(dict(scope=__doc__,checkpoint_sha256=freeze['checkpoint_sha256'],all_candidates_retained=True,selection='Presentation only after physics and visual inspection; no policy selection or retuning.'),indent=2))
 print(name,'complete',flush=True)
