"""Global development selection including source-disjoint expansion validation.
Rule fixed before expanded training outcomes; final tests never read here.
"""
import json
from pathlib import Path
D=Path('/home/pku/frankenmotion/outputs_amass/universal_tracker_20261006');base=json.loads((D/'development_selection_audit.json').read_text());records=[]
for row in base['results']:
 r=dict(row);s=D/'evaluation'/r['name']/'summary.json';n=D/'general_evaluation'/('natural_'+r['name'])/'fidelity_summary.json';r['expansion_selection_complete']=s.exists() and n.exists();r['expansion_eligible']=False
 if r['expansion_selection_complete']:
  eleven=json.loads(s.read_text());natural=json.loads(n.read_text())['results'];assert len(eleven)==11 and len(natural)==12;minimum_completion=min(v['actual']['measurable']/v['actual']['planned'] for v in eleven.values());joint=sum(v['actual']['joint_pass'] for v in eleven.values());r['minimum_class_completion']=minimum_completion;r['eleven_joint_passes']=joint;r['natural_accurate_complete']=sum(v['accurate_complete'] for v in natural.values());r['natural_requests']=sum(v['requests'] for v in natural.values());r['expansion_score']=(sum(v['actual']['joint_pass']/v['actual']['planned'] for v in eleven.values())+sum(v['accurate_complete']/v['requests'] for v in natural.values()))/23;r['expansion_eligible']=bool(r['table_and_smoothness_gate'] and joint>=49 and minimum_completion>=.8)
 records.append(r)
ranked=sorted([r for r in records if r['expansion_eligible']],key=lambda r:(-r['expansion_score'],r['eleven']['macro_semantic_E_all'],r['name']));result=dict(scope=__doc__,rule='Historical table6/6,jitter<=1.5stable; 11taskjoint>=oldbroad49/110 and perclasscompletion>=80%. Among eligible, maximize equal-category mean of11 numeric+event rates and12 naturalreference-accuratecompletion rates. These are different metrics, combined only for development model choice; final tables report them separately. Ties use lower11 semantic error. No perclass deployment selection.',rule_timing='Defined during development before expanded training starts; not a preregistration preceding all exploratory experiments.',results=records,ranking=[r['name'] for r in ranked]);(D/'expansion_selection_audit.json').write_text(json.dumps(result,indent=2));print('Eligible ranking',result['ranking'])
for r in ranked:print(r['name'],r['expansion_score'],r['eleven_joint_passes'],r['natural_accurate_complete'],r['natural_requests'])
