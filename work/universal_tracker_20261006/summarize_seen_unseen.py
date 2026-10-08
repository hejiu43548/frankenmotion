"""Matched RSI entry diagnostics, never a policy selection criterion."""
import json
from pathlib import Path
D=Path('/home/pku/frankenmotion/outputs_amass/universal_tracker_20261006');rows=[]
for name in ['training_reference_v4_3000','training_reference_v5_1000','training_reference_v5_3000','training_reference_v5_final','validation_rsi_v5_1000','validation_rsi_v5_3000','validation_rsi_v5_final']:
 p=D/'general_evaluation'/name/'fidelity_summary.json'
 if not p.exists():continue
 data=json.loads(p.read_text())['results'];rows.append(dict(name=name,**{key:sum(v[key] for v in data.values()) for key in ['requests','complete','accurate_complete']},per_task=data))
result=dict(scope=__doc__,caution='Training diagnostic samples and validation have different sources/composition. Same RSI entry removes one confound but does not isolate overfitting or establish a causal effect. Poor seen-source performance means failure cannot be attributed solely to unseen-source generalization.',results=rows)
(D/'seen_unseen_diagnostic.json').write_text(json.dumps(result,indent=2));print([(r['name'],r['requests'],r['complete'],r['accurate_complete']) for r in rows])
