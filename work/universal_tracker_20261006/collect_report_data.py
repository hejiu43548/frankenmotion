"""Compact final evidence, with explicit missing files and no imputed successes."""
import json,datetime,subprocess,sys
from pathlib import Path
D=Path('/home/pku/frankenmotion/outputs_amass/universal_tracker_20261006');missing=[]
def read(path,required=False):
 p=D/path
 if not p.exists():
  missing.append(str(path))
  if required:raise FileNotFoundError(p)
  return None
 return json.loads(p.read_text())
out=dict(created_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),frozen=read('frozen_unified/protocol.json',True),actor=read('frozen_unified/actor.json',True),plan=read('frozen_unified/final_test_plan.json',True),controllers={})
for tag in ['candidate','broad','stable']:
 name='fresh_'+tag;summary=read(f'general_evaluation/{name}/eleven_summary.json',True);audit=read(f'general_evaluation/{name}/eleven_audit.json',True)
 passes={task:bool(s['actual']['measurable']/s['actual']['planned']>=.9 and s['actual']['event_pass']/s['actual']['planned']>=.9 and s['actual']['joint_pass']/s['actual']['planned']>=.8) for task,s in summary.items()}
 out['controllers'][tag]=dict(aggregate=audit['aggregate'],per_task=summary,class_pass=passes,passed_classes=sum(passes.values()),table=read(f'table_evaluation/{name}/summary.json',True),natural=read(f'general_evaluation/{name}_natural/fidelity_summary.json',True))
out['sonic']=dict(audit=read('sonic_evaluation/fresh_final/audit.json'),per_task=read('sonic_evaluation/fresh_final/summary.json'),protocol=read('sonic_evaluation/fresh_final/protocol.json'))
tol=dict(raise_hand=.02,reach=.02,strike=.1,wave=.015,turn=.08,sidestep=.06,back_walk=.05,kick=.05,jump=.04,lean=.06,walk=.05)
rows=read('general_evaluation/fresh_candidate/audited_results.json',True);transitions={}
for task in tol:
 rr=[r for r in rows if r['task']==task];patterns={}
 for r in rr:
  bits=''.join('1' if r[k] is not None and r[k]['event_pass'] and abs(r[k]['quantity']-r['command'])<=tol[task] else '0' for k in ['human','g1','actual']);patterns[bits]=patterns.get(bits,0)+1
 transitions[task]=dict(requests=len(rr),patterns=patterns)
out['stage_preservation']=dict(order=['human','g1','actual'],definition='Bit1 means event+numeric tolerance pass. Descriptive paired transitions, not isolated causal attribution; actual may improve a biased reference by chance.',per_task=transitions)
out['statistics']={tag:read('fresh_statistics_'+tag+'.json',True) for tag in ['broad','stable']}
out['gpu']={tag:read(f'evaluation/fresh_gpu_{tag}/audit.json') for tag in ['candidate','broad']}
out['robustness']={tag:{profile:read(f'general_evaluation/fresh_robust_{tag}_{profile}/eleven_audit.json') for profile in ['nominal','friction_0p6','mass_1p1','delay_20ms','lateral_push_40N']} for tag in ['candidate','broad']}
out['development_selection']=read('frozen_unified/expansion_selection_audit.json',True);out['training_inventory']=read('release/inventory.json');out['single_policy_verification']=read('single_policy_verification.json')
subprocess.run([sys.executable,'/home/pku/frankenmotion/work/universal_tracker_20261006/audit_natural_termination.py','--run',str(D/'general_evaluation/fresh_candidate_natural')],check=True,stdout=subprocess.DEVNULL)
out['natural_termination']=read('general_evaluation/fresh_candidate_natural/termination_audit.json',True)
out['datasets']={name:read(path) for name,path in [('base','joint_corpus_v1/audit.json'),('expanded','joint_corpus_expanded_corrected/audit.json'),('natural_test','natural_test/protocol.json'),('fresh_generation','fresh_final/protocol.json'),('fresh_table','fresh_table/generation_protocol.json')]}
subprocess.run([sys.executable,'/home/pku/frankenmotion/work/universal_tracker_20261006/collect_table_quality.py'],check=True)
out['table_quality']=read('fresh_table_quality.json',True)
out['demos']={name:dict(summary=read(f'general_evaluation/frozen_demo_{name}/fidelity_summary.json'),commands=read(f'general_evaluation/frozen_demo_{name}/command_metrics.json') if name in ['navigation','service','service_trimmed','retreat'] else None) for name in ['point','navigation','official_text','service','service_trimmed','clean_bow','clean_bend','retreat']}
subprocess.run([sys.executable,str(Path('/home/pku/frankenmotion/work/universal_tracker_20261006/summarize_seen_unseen.py'))],check=True)
out['diagnostics']={name:read(path) for name,path in [('natural_paired','natural_paired_errors.json'),('natural_support','general_evaluation/natural_v4_3000/natural_support_diagnostic.json'),('support_projection','natural_support_projection_audit.json'),('seen_unseen','seen_unseen_diagnostic.json'),('root_observability','root_counterfactual_audit.json'),('ballistic_probe','general_evaluation/ballistic_v4_3000/eleven_audit.json'),('ballistic_scope','ballistic_reference_probe/protocol.json')]}
out['replication']=dict(plan=read('replication_plan.json'),status=read('replication_status.json'),protocol=read('replication_protocol.json'))
if out['replication']['status'] and out['replication']['status']['stage']=='complete':
 out['replication']['results']=dict(aggregate=read('general_evaluation/replica_seed6107_eleven/eleven_audit.json',True)['aggregate'],per_task=read('general_evaluation/replica_seed6107_eleven/eleven_summary.json',True),natural=read('general_evaluation/replica_seed6107_natural/fidelity_summary.json',True),table=read('table_evaluation/replica_seed6107/summary.json',True))
study=D/'rollout_projection_study';out['projection_study']=None
if (study/'protocol.json').exists():
 out['projection_study']={k:json.loads((study/(k+'.json')).read_text()) if (study/(k+'.json')).exists() else None for k in ['protocol','projection_audit','status','training_protocol','results']}
out['long_horizon_delay']=read('long_horizon_delay_check.json');out['long_horizon_ablation']=read('long_horizon_ablation.json');out['long_horizon']=read('general_evaluation/frozen_demo_long_horizon_v3/long_horizon_summary.json');out['missing_optional_files']=missing;(D/'report_data.json').write_text(json.dumps(out,indent=2));print('Collected report evidence; missing optional',missing)
