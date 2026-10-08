from pathlib import Path
import sys,json,hashlib,numpy as np
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/unified_commands_20261007';O=R/'outputs_amass/merged_report_20261008';O.mkdir(exist_ok=True);sys.path[:0]=[str(R/'work'),str(R/'outputs_amass/franken_eleven_20261003/code')]
import transfer as tr
from audit_results import native_metrics,stats,TASKS,RANGES,TOLS
model=tr.rt.load_model();height=tr.robot_height(model)
for name in ['baseline_final','unified_final']:
 rows=json.loads((D/'general_evaluation'/name/'audited_results.json').read_text());rows=[r for r in rows if r['kind'] in TASKS];assert len(rows)==880;aud=[]
 for r in rows:
  ref=np.load(r['reference_path'])['reference_qpos'];aud.append(dict(r,human=native_metrics(r),g1=tr.measure(tr.get_positions(model,ref),r['task'],height)))
 summary={}
 for i,t in enumerate(TASKS):
  rr=[r for r in aud if r['task']==t];assert len(rr)==80;span=RANGES[i][1]-RANGES[i][0];summary[t]={k:stats(rr,k,span,TOLS[i]) for k in ['human','g1','actual']}
  for k in ['human','g1','actual']:summary[t][k]['semantic_E_all']=float(np.mean([min(abs(r[k]['quantity']-r['command'])/span,1) if r[k] is not None and r[k]['event_pass'] else 1 for r in rr]))
 out=O/'general_evaluation'/name;out.mkdir(parents=True,exist_ok=True);(out/'audited_results.json').write_text(json.dumps(aud,indent=2,default=lambda x:x.item()));(out/'eleven_summary.json').write_text(json.dumps(summary,indent=2,default=lambda x:x.item()));print(name,{k:sum(v[k]['joint_pass'] for v in summary.values()) for k in ['human','g1','actual']},flush=True)
