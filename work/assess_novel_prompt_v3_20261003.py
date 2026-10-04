"""Separate paraphrase stress-test audit, not unseen-action generalization."""
import sys,os,json,csv,hashlib
from pathlib import Path
import numpy as np
B=Path('/home/pku/frankenmotion/outputs_amass/franken_eleven_20261003');N=B.parent/'franken_improve_20261003';sys.path.insert(0,str(B/'code'));os.environ['ELEVEN_OUT']=str(B)
from audit_results import native_metrics,stats,TASKS,RANGES,TOLS
import transfer as tr
read=lambda p:json.loads(Path(p).read_text());protocol=read(N/'frozen_integrated_v3/protocol.json');wm=read(N/'frozen_controllers_v1/weight_map.json')
for path,h in protocol['weight_hashes'].items():assert hashlib.sha256(Path(path).read_bytes()).hexdigest()==h
manifest=read(N/'novel_prompt_v3/manifest.json');primary=read(N/'novel_prompt_v3_sonic/results.json');bm=read(N/'novel_prompt_v3_beyondmimic/results.json');assert len(manifest)==len(primary)==110 and len(bm)==30
ix=lambda rows:{(r['source'],r['command_index']):r for r in rows};pr=ix(primary);br=ix(bm);assert len(pr)==110 and len(br)==30;m=tr.rt.load_model();height=tr.robot_height(m);result=[];largest=0.;errors=[]
for src in manifest:
 key=(src['source'],src['command_index']);r=pr[key];assert all(r[k]==src[k] for k in ['path','task','seed','command']);tid=TASKS.index(src['task']);si=int(src['source'].split('_s')[-1]);assert src['seed']==75028000+tid*100+si
 o={k:src[k] for k in ['task','source','seed','command','command_index']};o['human']=native_metrics(src);o['controller']=protocol['task_route'][src['task']];o['g1']=None;o['actual']=None
 if r.get('error'):errors.append(r['error']);result.append(o);continue
 ref=np.load(N/'novel_prompt_v3_sonic'/(Path(src['path']).stem+'_uniform.npz'))['reference_qpos'];o['g1']=tr.measure(tr.get_positions(m,ref),src['task'],height);err=abs(o['g1']['quantity']-r['g1']['quantity']);assert err<1e-8;largest=max(largest,err);want=1+np.arange(len(ref))*.05
 if o['controller']=='sonic':
  z=np.load(N/'novel_prompt_v3_sonic'/(Path(src['path']).stem+'_uniform.npz'));states=z['qpos'];ts=z['time_s'];assert np.allclose(ts,(np.arange(len(states))+1)*.02);fall=r['fall_time']
 else:
  r=br[key];assert r['input_path']==src['path'] and r['seed']==src['seed'] and r['command']==src['command']
  if r.get('error'):errors.append(r['error']);result.append(o);continue
  assert r['checkpoint']==wm[src['task']] and r['termination_criterion']=='physical' and r['initialization']=='standing + 1s entry';states=np.load(N/'novel_prompt_v3_beyondmimic'/(Path(src['path']).stem+'_actual.npz'))['qpos'];ts=np.arange(len(states))*.02;fall=r['termination_time']
 assert np.isfinite(states).all()
 if r['actual'] is not None:
  assert fall is None and ts[-1]>=want[-1]-1e-7;p=tr.get_positions(m,states);p=np.stack([np.interp(want,ts,x) for x in p.reshape(len(p),-1).T],1).reshape(len(ref),24,3);a=tr.measure(p,src['task'],height);err=abs(a['quantity']-r['actual']['quantity']);assert err<1e-8 and a['event_pass']==r['actual']['event_pass'];largest=max(largest,err);o['actual']=a
 else:assert fall is not None or ts[-1]<want[-1]-1e-7
 result.append(o)
assert len(result)==110;summary={}
for i,task in enumerate(TASKS):
 rows=[r for r in result if r['task']==task];assert len(rows)==10;summary[task]={k:stats(rows,k,RANGES[i][1]-RANGES[i][0],TOLS[i]) for k in ['human','g1','actual']}
out=N/'final_delivery_novel_v3';out.mkdir(exist_ok=True);(out/'audited_results.json').write_text(json.dumps(result,indent=2,default=lambda x:x.item()));(out/'summary.json').write_text(json.dumps(summary,indent=2));(out/'audit.json').write_text(json.dumps(dict(requests=110,physical_rollouts=110,raw_max_error=largest,recorded_errors=errors,scope='One new whole-body caption and paraphrased local phrases per task, 2 noises; same task_id and schedules. No unseen-action/free-instruction claim.'),indent=2));flat=[]
for r in result:
 q={k:r[k] for k in ['task','source','seed','command','command_index','controller']}
 for k in ['human','g1','actual']:q[k+'_Q']=r[k]['quantity'] if r[k] else None;q[k+'_event']=r[k]['event_pass'] if r[k] else None
 flat.append(q)
with (out/'requests.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=flat[0]);w.writeheader();w.writerows(flat)
print(json.dumps({t:s['actual'] for t,s in summary.items()},indent=2))
