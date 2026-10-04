"""Compare historical and corrected BM runs without hiding failures."""
from pathlib import Path
import argparse,json,re
import numpy as np
N=Path('/home/pku/frankenmotion/outputs_amass/franken_improve_20261003');W=N.parent.parent/'work'
a=argparse.ArgumentParser();a.add_argument('--scope',choices=['integrated','v1','novel'],required=True);args=a.parse_args()
folders={'integrated':('integrated_v3_beyondmimic','integrated_v3_beyondmimic_capacity256','mjlab_probe_log.txt','capacity_integrated_v3.log',240),'v1':('beyondmimic_confirmation','beyondmimic_confirmation_capacity256','beyondmimic_confirmation_20261003.log','capacity_v1.log',880),'novel':('novel_prompt_v3_beyondmimic','novel_prompt_v3_beyondmimic_capacity256','bm_novel_prompt_v3.log','capacity_novel_v3.log',30)}
old,new,oldlog,newlog,n=folders[args.scope];before=json.loads((N/old/'results.json').read_text());after=json.loads((N/new/'results.json').read_text());assert len(before)==len(after)==n
key=lambda r:(r['source'],r['command_index']);bi={key(r):r for r in before};ai={key(r):r for r in after};assert len(bi)==len(ai)==n and bi.keys()==ai.keys();log=(W/oldlog).read_text();clean=(W/newlog).read_text();assert 'overflow' not in clean.lower();last=0;warnings={}
for line in log.splitlines():
 m=re.fullmatch(r'(\d+) requests complete',line.strip())
 if m:last=int(m[1])
 if 'overflow' in line.lower():warnings[last]=warnings.get(last,0)+1
rows=[]
for k,b in bi.items():
 c=ai[k];assert all(b[x]==c[x] for x in ['task','seed','command','input_path']);qb=b['actual']['quantity'] if b.get('actual') else None;qc=c['actual']['quantity'] if c.get('actual') else None;rows.append(dict(task=b['task'],source=b['source'],command_index=b['command_index'],command=b['command'],old_Q=qb,corrected_Q=qc,old_event=b['actual']['event_pass'] if b.get('actual') else False,corrected_event=c['actual']['event_pass'] if c.get('actual') else False,delta_Q=None if qb is None or qc is None else qc-qb))
summary={}
for t in sorted({r['task'] for r in rows}):
 rs=[r for r in rows if r['task']==t];d=[r['delta_Q'] for r in rs if r['delta_Q'] is not None];summary[t]=dict(planned=len(rs),old_complete=sum(r['old_Q'] is not None for r in rs),corrected_complete=sum(r['corrected_Q'] is not None for r in rs),old_event=sum(r['old_event'] for r in rs),corrected_event=sum(r['corrected_event'] for r in rs),completion_changed=sum((r['old_Q'] is None)!=(r['corrected_Q'] is None) for r in rs),event_changed=sum(r['old_event']!=r['corrected_event'] for r in rs),mean_abs_Q_change=float(np.mean(np.abs(d))) if d else None,max_abs_Q_change=float(np.max(np.abs(d))) if d else None)
report=dict(scope=args.scope,old_warning_lines=sum(warnings.values()),corrected_warning_lines=0,warning_source_attribution='Approximate: GPU warning stdout between request completion counters is assigned to the next request; retained for traceability, not exact causal proof.',warning_requests=[dict(source=before[min(i,n-1)]['source'],command_index=before[min(i,n-1)]['command_index'],warnings=count) for i,count in warnings.items()],summary=summary,rows=rows,note='Rerun differences may also include non-bitwise GPU physics variation; do not attribute every numerical delta solely to lost contacts. Frozen weights, inputs and thresholds unchanged.')
(N/('capacity_impact_'+args.scope+'.json')).write_text(json.dumps(report,indent=2));print(json.dumps(dict(scope=args.scope,old_warning_lines=report['old_warning_lines'],summary=summary),indent=2))
