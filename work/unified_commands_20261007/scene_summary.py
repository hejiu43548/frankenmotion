from pathlib import Path
import json,numpy as np
D=Path('/home/pku/frankenmotion/outputs_amass/unified_commands_20261007');out=D/'report';out.mkdir(exist_ok=True);res={}
for name in ['baseline_dev','feature_v3_dev','feature_v4_dev','baseline_final','unified_final']:
 file=D/'table'/name/'composed/shared_tracker_summary.json'
 if not file.exists():continue
 z=json.loads(file.read_text());rr=z['results'];entry={k:z[k] for k in ['planned','processed','successes','physical_complete','checkpoint_sha256']};entry['metrics']={k:float(np.mean([r['command_metrics'][k] for r in rr])) for k in ['actual_turn_deg','actual_away_displacement_m','actual_hold_wrist_forward_human_equiv_m']};entry['rows']=[dict(index=r['scene']['index'],success=r['success'],command=r['scene']['reach_command_human_m'],turn_command_deg=r['scene']['turn_command_deg'],direction_rad=r['scene']['direction_rad'],distance_robot_m=r['scene']['distance_robot_m'],metrics=r['command_metrics']) for r in rr];entry['reach_commands']={str(c):dict(mean=float(np.mean([r['command_metrics']['actual_hold_wrist_forward_human_equiv_m'] for r in rr if abs(r['scene']['reach_command_human_m']-c)<1e-6])),count=sum(abs(r['scene']['reach_command_human_m']-c)<1e-6 for r in rr)) for c in [.3,.5]};res[name]=entry
(out/'scene_summary.json').write_text(json.dumps(res,indent=2));print(json.dumps({k:{q:v[q] for q in ['successes','planned','reach_commands']} for k,v in res.items()},indent=2))
