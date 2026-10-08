"""Development-only checkpoint comparison; never reads fresh frozen test results."""
import json
from pathlib import Path
import numpy as np
D=Path('/home/pku/frankenmotion/outputs_amass/universal_tracker_20261006');rows=[]
for folder in sorted((D/'table_evaluation').iterdir()):
 if not (folder/'summary.json').exists():continue
 s=json.loads((folder/'summary.json').read_text())
 if not s.get('scope','').startswith('Historical development'):continue
 if s['processed']!=s['planned']:continue
 row=dict(name=folder.name,table_success=s['success'],table_planned=s['planned'],checkpoint_sha256=s['checkpoint_sha256'])
 for phase in ['walk','away']:
  metrics=[json.loads(f.read_text()) for f in folder.glob('scene_*/rollout/'+phase+'_jitter_metrics.json')]
  for key in ['torso_angular_velocity','joint_velocity','action']:
   row[phase+'_'+key+'_highpass_rms']=float(np.mean([m[key]['highpass_5hz_rms'] for m in metrics])) if metrics else None
 audit=D/'evaluation'/folder.name/'audit.json'
 if folder.name=='baseline_stable':audit=D/'evaluation/baseline_stable_dev/audit.json'
 if folder.name=='baseline_broad':audit=D/'evaluation/baseline_broad_dev/audit.json'
 if audit.exists():row['eleven']=json.loads(audit.read_text())['aggregate']
 rows.append(row)
baseline=next(r for r in rows if r['name']=='baseline_stable')
for r in rows:
 r['walk_torso_jitter_ratio']=r['walk_torso_angular_velocity_highpass_rms']/baseline['walk_torso_angular_velocity_highpass_rms'] if r['walk_torso_angular_velocity_highpass_rms'] is not None else None
 r['table_and_smoothness_gate']=r['table_planned']==6 and r['table_success']==6 and r['walk_torso_jitter_ratio'] is not None and r['walk_torso_jitter_ratio']<=1.5
(D/'development_selection_audit.json').write_text(json.dumps(dict(scope='Historical development only; table6/6 and torso highpass<=1.5 stable are necessary, not sufficient. No task-dependent checkpoint selection. Existing11 per-class semantic thresholds remain unchanged.',results=rows),indent=2))
for r in rows:print(r['name'],'table',r['table_success'],'jitter_ratio',round(r['walk_torso_jitter_ratio'] or 0,3),'eleven',r.get('eleven',{}))
