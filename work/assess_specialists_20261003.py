"""Apply predeclared development gate; do not freeze or claim holdout success."""
from pathlib import Path
import json,sys,os,hashlib
import numpy as np
R=Path('/home/pku/frankenmotion');N=R/'outputs_amass/franken_improve_20261003';B=R/'outputs_amass/franken_eleven_20261003';sys.path.insert(0,str(B/'code'));os.environ['ELEVEN_OUT']=str(B)
import transfer as tr
read=lambda p:json.loads(p.read_text())
rule=read(N/'specialist_selection_rule.json');baseline=read(N/'physical_development_results.json');m=tr.rt.load_model();height=tr.robot_height(m);report={};maxerr=0.
for task,span,suffix in [('turn',1.05,'ori2.0_vel1.0'),('strike',1.,'ori0.5_vel2.0')]:
 base=sorted([r for r in baseline if r['task']==task and r['variant']=='uniform'],key=lambda r:r['command']);assert len(base)==5
 groups={}
 for label,folder in [('specialist','specialist_'+task+'_development_capacity256'),('dance','specialist_dance_control_capacity256')]:
  path=N/folder;rows=sorted([r for r in read(path/'results.json') if r['task']==task],key=lambda r:r['command']);assert len(rows)==5
  assert 'overflow' not in (R/'work'/(folder+'.log')).read_text().lower()
  contract=read(path/'contract.json');assert contract['contact_capacity_nconmax']==256 and contract['constraint_capacity_njmax']==2048
  for r,b in zip(rows,base):
   assert all(r[k]==b[k] for k in ['task','source','seed','command','command_index']) and r['input_path']==b['path'];assert not r.get('error')
   if label=='specialist':assert r['checkpoint']==str(N/('beyondmimic_finetune_'+task+'_all_'+suffix)/'model_2399.pt')
   stem=Path(b['path']).stem;states=np.load(path/(stem+'_actual.npz'))['qpos'];ref=np.load(N/'physical_development_eval/gmr_probe'/(stem+'_uniform.npz'))['reference_qpos'];ts=np.arange(len(states))*.02;want=1+np.arange(len(ref))*.05;assert np.isfinite(states).all()
   if r['actual'] is not None:
    assert r['termination_time'] is None and ts[-1]>=want[-1]-1e-7 and len(states)==r['target_steps'];p=tr.get_positions(m,states);p=np.stack([np.interp(want,ts,x) for x in p.reshape(len(p),-1).T],1).reshape(len(ref),24,3);a=tr.measure(p,task,height);err=abs(a['quantity']-r['actual']['quantity']);assert err<1e-8 and a['event_pass']==r['actual']['event_pass'];maxerr=max(maxerr,err)
   else:assert r['termination_time'] is not None or ts[-1]<want[-1]-1e-7
  groups[label]=rows
 groups['sonic']=base
 def metrics(rows):
  q=[r['actual']['quantity'] if r['actual'] is not None else None for r in rows];events=[r['actual'] is not None and bool(r['actual']['event_pass']) for r in rows]
  return dict(Q=q,complete=sum(x is not None for x in q),events=sum(events),monotonic=all(x is not None for x in q) and all(q[i+1]>=q[i]-1e-6 for i in range(4)),semantic_E_all=float(np.mean([min(abs(x-r['command'])/span,1.) if e else 1. for r,x,e in zip(rows,q,events)])))
 stats={k:metrics(v) for k,v in groups.items()};a=stats['specialist'];delta=stats['sonic']['semantic_E_all']-a['semantic_E_all'];selected=a['complete']==5 and a['events']==5 and a['monotonic'] and delta>.02;report[task]=dict(commands=[r['command'] for r in base],metrics=stats,semantic_improvement=delta,passes_preregistered_gate=selected)
out=N/'final_delivery_specialists';out.mkdir(exist_ok=True);data=dict(rule=rule,raw_recalculation_max_error=maxerr,results=report,scope='Development only. Fresh frozen holdout required before adoption.');(out/'selection.json').write_text(json.dumps(data,indent=2));print(json.dumps(data,indent=2))
