"""Run locked evaluation; never train, select or alter a candidate after starting."""
import sys,json,subprocess,hashlib
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/jump_tracker_20261007';W=R/'work/jump_tracker_20261007';py=sys.executable
freeze=D/'candidate_freeze.json';frozen_bytes=freeze.read_bytes();selection=json.loads(frozen_bytes);actor=Path(selection['checkpoint']);assert hashlib.sha256(actor.read_bytes()).hexdigest()==selection['checkpoint_sha256']
subprocess.run([str(R/'.conda/bin/python'),str(W/'generate_fresh.py')],check=True)
subprocess.run([py,str(W/'prepare_fresh.py')],check=True)
for name,ck,ac in [('final_baseline',D/'backup/policy.pt',D/'backup/actor.pt'),('final_candidate',actor,actor)]:
 subprocess.run([py,str(W/'evaluate_actor.py'),'--checkpoint',str(ck),'--actor',str(ac),'--manifest',str(D/'fresh_final/native_manifest.json'),'--name',name,'--workers','4'],check=True)
 subprocess.run([py,str(W/'assess.py'),'--name',name],check=True)
 subprocess.run([py,str(W/'audit_contacts.py'),'--run',str(D/'general_evaluation'/name)],check=True)
 subprocess.run([py,str(W/'diagnose_final.py'),'--run',str(D/'general_evaluation'/name),'--output',str(D/'diagnostics'/name)],check=True)
for label,ck,ac,profile in [('baseline_delay',D/'backup/policy.pt',D/'backup/actor.pt','delay_20ms'),('baseline_predict',D/'backup/policy.pt',D/'backup/actor.pt','delay_predict'),('candidate_delay',actor,actor,'delay_20ms'),('candidate_predict',actor,actor,'delay_predict'),('candidate_predict_mass',actor,actor,'delay_predict_mass_1p1')]:
 name='final_'+label;script='evaluate_predictive.py' if profile.startswith('delay_predict') else 'evaluate_robust_actor.py'
 subprocess.run([py,str(W/script),'--checkpoint',str(ck),'--actor',str(ac),'--profile',profile,'--manifest',str(D/'fresh_final/standard_manifest.json'),'--name',name,'--workers','4'],check=True)
 subprocess.run([py,str(W/'assess.py'),'--name',name],check=True)
assert freeze.read_bytes()==frozen_bytes and hashlib.sha256(actor.read_bytes()).hexdigest()==selection['checkpoint_sha256']
subprocess.run([py,str(W/'render_comparison.py'),'--baseline',str(D/'general_evaluation/final_baseline'),'--candidate',str(D/'general_evaluation/final_candidate'),'--output',str(D/'visuals/fresh_fixed_comparison')],check=True)
(D/'final_pipeline_complete.json').write_text(json.dumps(dict(candidate_sha256=selection['checkpoint_sha256'],freeze_sha256=hashlib.sha256(frozen_bytes).hexdigest(),primary_requests=60,secondary_robustness_requests_per_condition=40),indent=2))
