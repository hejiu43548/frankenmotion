"""Audit 1120 unique physical executions and assess the integrated 880-request suite."""
import sys,os,json,csv,hashlib
from pathlib import Path
import numpy as np
BASE=Path('/home/pku/frankenmotion/outputs_amass/franken_eleven_20261003');ROOT=BASE.parent/'franken_improve_20261003';PROJECT=ROOT.parent.parent;sys.path.insert(0,str(BASE/'code'));os.environ['ELEVEN_OUT']=str(BASE)
from audit_results import native_metrics,stats,TASKS,RANGES,TOLS
import transfer as tr
CAP_FOLDER='integrated_v3_beyondmimic_capacity256';CAP_LOG='capacity_integrated_v3.log'
# Capacity correction changes allocation only; keep model selection frozen.
_cap_protocol=json.loads((ROOT/'capacity_correction_protocol.json').read_text())
assert __import__('hashlib').sha256(Path('/home/pku/frankenmotion/work/mjlab_probe_capacity_20261003.py').read_bytes()).hexdigest()==_cap_protocol['script_sha256']
_cap_contract=json.loads((ROOT/CAP_FOLDER/'contract.json').read_text())
assert _cap_contract['contact_capacity_nconmax']==256 and _cap_contract['constraint_capacity_njmax']==2048
assert 'overflow' not in Path('/home/pku/frankenmotion/work/'+CAP_LOG).read_text().lower()

read=lambda p:json.loads(Path(p).read_text());sha=lambda p:hashlib.sha256(Path(p).read_bytes()).hexdigest()
protocol=read(ROOT/'frozen_integrated_v3/protocol.json')
for path,digest in {**protocol['code_hashes'],**protocol['weight_hashes']}.items():assert sha(path)==digest,path
wm=read(ROOT/'frozen_controllers_v1/weight_map.json');cp=read(ROOT/'frozen_controllers_v1/protocol.json')
for r in cp['weights'].values():assert sha(r['frozen'])==r['sha256']
def index(rows,n):
 d={(r['source'],r['command_index']):r for r in rows};assert len(d)==len(rows)==n;return d
manifest=index(read(ROOT/'integrated_v3_manifest.json'),880);control_manifest=index(read(ROOT/'integrated_v3_control_manifest.json'),240)
primary=index(read(ROOT/'integrated_v3_sonic/results.json'),880);control=index(read(ROOT/'confirmation/integrated_v3_control/results.json'),240);bm=index(read(ROOT/'integrated_v3_beyondmimic_capacity256/results.json'),240)
m=tr.rt.load_model();height=tr.robot_height(m);largest=0.;checks=[];verified_generation=set()
def remeasure(row,folder,kind,ref):
 global largest
 if row.get('error'):checks.append(dict(source=row['source'],command_index=row['command_index'],kind=kind,status='recorded_error',error=row['error']));return None
 if kind=='bm':
  assert row['checkpoint']==wm[row['task']];assert row['termination_criterion']=='physical' and row['initialization']=='standing + 1s entry';stem=Path(row['input_path']).stem;z=np.load(folder/(stem+'_actual.npz'));states=z['qpos'];ts=np.arange(len(states))*.02;fall=row['termination_time']
 else:
  stem=Path(row['path']).stem;z=np.load(folder/(stem+'_uniform.npz'));states=z['qpos'];ts=z['time_s'];assert np.allclose(ts,(np.arange(len(states))+1)*.02,atol=1e-10);fall=row['fall_time']
 assert np.isfinite(states).all();want=1+np.arange(len(ref))*.05;actual=None
 if row.get('actual') is not None:
  assert fall is None and ts[-1]>=want[-1]-1e-7
  if kind=='bm':assert len(states)==row['target_steps']
  p=tr.get_positions(m,states);p=np.stack([np.interp(want,ts,x) for x in p.reshape(len(p),-1).T],1).reshape(len(ref),24,3);actual=tr.measure(p,row['task'],height);err=abs(actual['quantity']-row['actual']['quantity']);assert err<1e-8 and actual['event_pass']==row['actual']['event_pass'];largest=max(largest,err)
 else:assert fall is not None or ts[-1]<want[-1]-1e-7
 checks.append(dict(source=row['source'],command_index=row['command_index'],kind=kind,status='verified'));return actual
rows=[]
for k,src in manifest.items():
 task=src['task'];pi=int(src['source'].split('_p')[-1].split('_')[0]);si=int(src['source'].split('_s')[-1]);assert src['seed']==64028000+pi*100+si
 expected=ROOT/protocol['generation'].get(task,protocol['generation']['all_other_tasks']);provpath=Path(src['path']).parent/'provenance.json'
 if str(provpath) not in verified_generation:
  gp=read(provpath);assert gp['sha256']==sha(expected);assert Path(gp['weights']).resolve()==expected.resolve();verified_generation.add(str(provpath))
 r=primary[k];assert all(r[x]==src[x] for x in ['path','task','seed','command']);o={x:src[x] for x in ['task','source','seed','command','command_index']};o['human']=native_metrics(src);o['controller']=protocol['task_route'][task];o['baseline_shared_execution']=task not in ['raise_hand','lean','kick']
 if r.get('error'):
  ref=None;o['g1']=None;o['actual']=None;checks.append(dict(source=src['source'],command_index=src['command_index'],kind=o['controller'],status='recorded_error',error=r['error']))
 else:
  ref=np.load(ROOT/'integrated_v3_sonic'/(Path(src['path']).stem+'_uniform.npz'))['reference_qpos'];g1=tr.measure(tr.get_positions(m,ref),task,height);err=abs(g1['quantity']-r['g1']['quantity']);assert err<1e-8 and g1['event_pass']==r['g1']['event_pass'];largest=max(largest,err);o['g1']=g1
  if o['controller']=='sonic':o['actual']=remeasure(r,ROOT/'integrated_v3_sonic','sonic',ref)
  else:
   b=bm[k];assert b['input_path']==src['path'] and b['seed']==src['seed'] and b['command']==src['command'];o['actual']=remeasure(b,ROOT/'integrated_v3_beyondmimic_capacity256','bm',ref)
 if o['baseline_shared_execution']:o['v1_actual']=o['actual'];o['v1_human']=o['human'];o['v1_g1']=o['g1']
 else:
  c=control[k];cs=control_manifest[k];assert all(c[x]==cs[x] for x in ['path','task','seed','command']);assert all(cs[x]==src[x] for x in ['task','seed','command']);o['v1_human']=native_metrics(cs)
  if c.get('error'):o['v1_g1']=None;o['v1_actual']=remeasure(c,ROOT/'confirmation/integrated_v3_control','control',None)
  else:
   cr=np.load(ROOT/'confirmation/integrated_v3_control'/(Path(c['path']).stem+'_uniform.npz'))['reference_qpos'];o['v1_g1']=tr.measure(tr.get_positions(m,cr),task,height);assert abs(o['v1_g1']['quantity']-c['g1']['quantity'])<1e-8;o['v1_actual']=remeasure(c,ROOT/'confirmation/integrated_v3_control','control',cr)
 rows.append(o)
assert len(checks)==1120 and sum(r['baseline_shared_execution'] for r in rows)==640
summary={};rng=np.random.default_rng(64028);keys=['human','g1','actual','v1_human','v1_g1','v1_actual']
for i,task in enumerate(TASKS):
 a=[r for r in rows if r['task']==task];assert len(a)==80;span=RANGES[i][1]-RANGES[i][0];summary[task]={key:stats(a,key,span,TOLS[i]) for key in keys}
 for key in keys:
  summary[task][key]['semantic_E_all']=float(np.mean([min(abs(r[key]['quantity']-r['command'])/span,1) if r[key] is not None and r[key]['event_pass'] else 1 for r in a]));complete=[]
  for src in sorted({r['source'] for r in a}):
   group=sorted([r for r in a if r['source']==src],key=lambda r:r['command'])
   if all(r[key] is not None for r in group):complete.append(all(group[j+1][key]['quantity']>=group[j][key]['quantity']-1e-6 for j in range(4)))
  summary[task][key]['monotonic_complete_sources']=sum(complete);summary[task][key]['monotonic_sources_measurable']=len(complete)
 d=[]
 for src in sorted({r['source'] for r in a}):
  g=[r for r in a if r['source']==src];err=lambda k:np.mean([min(abs(r[k]['quantity']-r['command'])/span,1) if r[k] is not None else 1 for r in g]);d.append(err('actual')-err('v1_actual'))
 b=np.mean(np.array(d)[rng.integers(0,16,(5000,16))],axis=1);summary[task]['paired_E_all_delta']=dict(mean=float(np.mean(d)),source_bootstrap95=np.quantile(b,[.025,.975]).tolist(),unchanged_shared_execution=task not in ['raise_hand','lean','kick'])
out=ROOT/'final_delivery_integrated_v3_capacity256';out.mkdir(exist_ok=True);(out/'audited_results.json').write_text(json.dumps(rows,indent=2,default=lambda x:x.item()));(out/'summary.json').write_text(json.dumps(summary,indent=2));audit=dict(contact_capacity=256,constraint_capacity=2048,overflow_warnings=0,requests=880,unique_physical_rollouts=1120,checks=len(checks),shared_unchanged_request_executions=640,raw_recalc_max_error=largest,recorded_errors=sum(r['status']=='recorded_error' for r in checks),frozen_hashes_verified=True,checks_detail=checks);(out/'audit.json').write_text(json.dumps(audit,indent=2))
flat=[]
for r in rows:
 line={k:r[k] for k in ['task','source','seed','command','command_index','controller','baseline_shared_execution']}
 for key in keys:
  line[key+'_Q']=r[key]['quantity'] if r[key] else None;line[key+'_event']=r[key]['event_pass'] if r[key] else None
  if r[key]:
   for field,value in r[key].items():
    if field not in ['quantity','event_pass'] and isinstance(value,(str,int,float,bool)):line[key+'_'+field]=value
 flat.append(line)
with (out/'requests.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in flat for k in r)));w.writeheader();w.writerows(flat)
print(json.dumps({k:v for k,v in audit.items() if k!='checks_detail'}))
