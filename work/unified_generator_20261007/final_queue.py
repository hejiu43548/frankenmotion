"""Run fresh paired evaluation only AFTER an immutable selection record exists."""
import json,subprocess,time,hashlib
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/unified_generator_20261007';W=R/'work/unified_generator_20261007';py=R/'.conda/bin/python';ap=R/'work/mjlab_stable_env/bin/python'
s=json.loads((D/'final_selection.json').read_text());ck=Path(s['checkpoint']);assert hashlib.sha256(ck.read_bytes()).hexdigest()==s['sha256'];actor=D/'backup/tracker_candidate_actor.pt'
def run(args,log):
 with (D/log).open('w') as f:subprocess.run([str(v) for v in args],cwd=R,stdout=f,stderr=subprocess.STDOUT,check=True)
run([py,W/'export_full.py','--checkpoint',ck,'--name','selected_final'],'final_export.log')
full=D/'exports/selected_final/unified_generator.pt'
for name,extra in [('teacher_final',['--teachers']),('unified_final',['--checkpoint',str(full),'--full'])]:
 run([py,W/'generate_eval.py','--name',name,'--split','final']+extra,name+'_generation.log')
 run([ap,W/'assess.py','--name',name],name+'_audit.log')
 run([ap,W/'prepare_tracking.py','--name',name],name+'_retarget.log')
 run([ap,W/'evaluate_tracker.py','--checkpoint',actor,'--actor',actor,'--manifest',D/'generation'/name/'native_manifest.json','--name',name,'--workers','4'],name+'_tracking.log')
 run([ap,W/'assess_tracker.py','--name',name],name+'_tracking_audit.log')
(D/'final_evaluation_complete.json').write_text(json.dumps(dict(completed_at=time.time(),selection=s,protocol='All 880 paired samples on each generator and same frozen tracker; no final-set training.')))
