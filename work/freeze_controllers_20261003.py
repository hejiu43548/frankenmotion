"""Choose only from DEVELOPMENT evidence, then freeze before unsealing confirmation."""
from pathlib import Path
import json,hashlib,shutil,datetime
ROOT=Path('/home/pku/frankenmotion/outputs_amass/franken_improve_20261003');OUT=ROOT/'frozen_controllers_v1'
assert not (OUT/'protocol.json').exists(),'Selection already frozen; do not overwrite'
TASKS=['raise_hand','reach','strike','wave','turn','sidestep','back_walk','kick','jump','lean','walk'];SPANS=[.4,.3,1.,.14,1.05,.8,.55,.45,.3,.5,.6]
def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def scores(rows,task):
 rows=[r for r in rows if r['task']==task];assert len(rows)==5,(task,len(rows));span=SPANS[TASKS.index(task)];actual=[r.get('actual') for r in rows];err=[min(abs(v['quantity']-r['command'])/span,1) if v else 1 for r,v in zip(rows,actual)];event=[bool(v and v['event_pass']) for v in actual];return dict(semantic_E_all=sum(e if ok else 1 for e,ok in zip(err,event))/5,E_all=sum(err)/5,complete=sum(v is not None for v in actual),event=sum(event))
sonic=[r for r in read(ROOT/'physical_development_results.json') if r['variant']=='uniform' and r['task']!='jump']+[r for r in read(ROOT/'jump_aligned_development_results.json') if r['variant']=='uniform']
sources={
 'dance':(Path('/home/pku/frankenmotion/work/beyondmimic_demo.pt'),ROOT/'beyondmimic_dance_common_condition/results.json'),
 'back_multi':(ROOT/'beyondmimic_finetune_back_walk_root2.0_all/model_1199.pt',ROOT/'beyondmimic_back_multi_common/results.json'),
 'locomotion':(ROOT/'beyondmimic_finetune_locomotion_root2.0_all/model_2399.pt',ROOT/'beyondmimic_locomotion_development/results.json'),
 'side_std03':(ROOT/'beyondmimic_finetune_sidestep_root2.0_all/model_2399.pt',ROOT/'beyondmimic_side_std03_development/results.json'),
 'side_std01':(ROOT/'beyondmimic_finetune_sidestep_root2.0_all_std0.1/model_2399.pt',ROOT/'beyondmimic_side_std01_development/results.json')}
loaded={k:read(p[1]) for k,p in sources.items()};jump=read(ROOT/'beyondmimic_jump_aligned_common/results.json');decisions={};route={};bank={}
for task in TASKS:
 options=['dance']+(['back_multi','locomotion'] if task=='back_walk' else ['side_std03','side_std01','locomotion'] if task=='sidestep' else ['locomotion'] if task=='walk' else [])
 stats={k:scores(jump if task=='jump' and k=='dance' else loaded[k],task) for k in options};best=min(options,key=lambda k:(stats[k]['semantic_E_all'],stats[k]['E_all']));ss=scores(sonic,task);route[task]='beyondmimic' if stats[best]['semantic_E_all']+.02<ss['semantic_E_all'] else 'sonic';bank[task]=best;decisions[task]=dict(sonic=ss,beyondmimic_options=stats,beyondmimic_selected=best,primary_route=route[task])
OUT.mkdir(exist_ok=True);frozen={}
for name in set(bank.values())|{'dance'}:
 src=sources[name][0];dest=OUT/(name+'.pt');digest=sha(src)
 if dest.exists():assert sha(dest)==digest
 else:shutil.copy2(src,dest)
 frozen[name]=dict(source=str(src),frozen=str(dest),sha256=digest)
weight_map={task:frozen[name]['frozen'] for task,name in bank.items()};(OUT/'weight_map.json').write_text(json.dumps(weight_map,indent=2))
record=dict(frozen_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),selection_data='Development only: one prompt/noise pair per task, five commands. Confirmation metrics not inspected.',selection_rule='Minimize semantic_E_all (event failure or missing=1, otherwise clipped normalized numeric error); use BeyondMimic in primary route only when improvement over SONIC exceeds 0.02. Per-task BM bank minimizes same score, numeric E_all breaks ties.',task_route=route,beyondmimic_weight_bank=bank,weights=frozen,decisions=decisions,generation_protocol_sha256=sha(ROOT/'frozen_generation_v1/protocol.json'),evaluation='Standing + 1s transition; physical fall root height<0.35m or norm(roll,pitch)>pi/3; native actuator/simulator differences disclosed.',primary_scope='Task-routed offline motion execution, not seamless online policy switching or hardware deployment')
(OUT/'protocol.json').write_text(json.dumps(record,indent=2));print(json.dumps(dict(task_route=route,beyondmimic_bank=bank),indent=2))
