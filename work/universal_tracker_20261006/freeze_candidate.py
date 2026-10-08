"""Freeze a single policy using completed development audits only."""
import json,hashlib,shutil,subprocess,sys,datetime
from pathlib import Path
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/universal_tracker_20261006';W=R/'work/universal_tracker_20261006'
sys.path.insert(0,str(R/'work'))
from unified_preview_20261004 import checkpoint_offsets
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
for script in ['selection_audit.py','selection_expansion.py']:
 subprocess.run([sys.executable,str(W/script)],check=True)
selection=json.loads((D/'expansion_selection_audit.json').read_text());assert selection['ranking'],'No globally eligible checkpoint'
name=selection['ranking'][0];evaluation=D/'evaluation'/name/'protocol.json';e=json.loads(evaluation.read_text());source=Path(e['checkpoint']);assert sha(source)==e['sha256']
selected=next(r for r in selection['results'] if r['name']==name);assert selected['expansion_eligible']
out=D/'frozen_unified';out.mkdir(exist_ok=False);shutil.copy2(source,out/'policy.pt');checkpoint=out/'policy.pt'
for filename in ['final_test_plan.json','expansion_selection_audit.json','development_selection_audit.json']:
 shutil.copy2(D/filename,out/filename)
subprocess.run([sys.executable,str(R/'work/unified_export_actor_20261004.py'),'--checkpoint',str(checkpoint),'--output',str(out/'actor.pt')],check=True)
protocol=dict(frozen_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),selected=name,source_checkpoint=str(source),checkpoint=str(checkpoint),checkpoint_sha256=sha(checkpoint),actor_sha256=sha(out/'actor.pt'),preview=True,preview_offsets=list(checkpoint_offsets(checkpoint)),task_routing=False,selection=selected,selection_rule=selection['rule'],selection_rule_timing=selection['rule_timing'],selection_audit_sha256=sha(out/'expansion_selection_audit.json'),final_test_plan_sha256=sha(out/'final_test_plan.json'),no_retune_after_freeze=True,scope='One shared tracker; generator still uses existing task/control adapters. Inherits BeyondMimic training and historical SONIC teacher lineage; includes frozen stable-actor replay retention loss. Not a claim of all11 passing or zero distillation.')
(out/'protocol.json').write_text(json.dumps(protocol,indent=2));print(json.dumps(protocol,indent=2))
