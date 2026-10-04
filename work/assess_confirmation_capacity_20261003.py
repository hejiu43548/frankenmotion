"""Paired held-out-noise assessment. Only run after controller selection is frozen."""
import os,sys,json,csv
from pathlib import Path
import numpy as np
BASE=Path('/home/pku/frankenmotion/outputs_amass/franken_eleven_20261003');ROOT=BASE.parent/'franken_improve_20261003';sys.path.insert(0,str(BASE/'code'));os.environ['ELEVEN_OUT']=str(BASE)
from audit_results import native_metrics,stats,TASKS,RANGES,TOLS
CAP_FOLDER='beyondmimic_confirmation_capacity256';CAP_LOG='capacity_v1.log'
# Capacity correction changes allocation only; keep model selection frozen.
_cap_protocol=json.loads((ROOT/'capacity_correction_protocol.json').read_text())
assert __import__('hashlib').sha256(Path('/home/pku/frankenmotion/work/mjlab_probe_capacity_20261003.py').read_bytes()).hexdigest()==_cap_protocol['script_sha256']
_cap_contract=json.loads((ROOT/CAP_FOLDER/'contract.json').read_text())
assert _cap_contract['contact_capacity_nconmax']==256 and _cap_contract['constraint_capacity_njmax']==2048
assert 'overflow' not in Path('/home/pku/frankenmotion/work/'+CAP_LOG).read_text().lower()

import protocol_audit_capacity_20261003  # Revalidate corrected request/checkpoint lineage.
freeze=ROOT/'frozen_controllers_v1/protocol.json';assert freeze.exists(),'Freeze controllers before inspecting confirmation'
protocol=json.loads(freeze.read_text());out=ROOT/'final_delivery_capacity256';out.mkdir(exist_ok=True)
def indexed(path,method=None):
 a=json.loads(Path(path).read_text());a=[r for r in a if method is None or r.get('method')==method];d={(r['source'],r['command_index']):r for r in a};assert len(d)==len(a)==880,(path,len(a));return d
bm_manifest=indexed(ROOT/'paired_baseline_confirmation_manifest.json');cm_manifest=indexed(ROOT/'candidate_confirmation_manifest.json');bd=indexed(ROOT/'confirmation/paired_baseline/results.json','direction');bg=indexed(ROOT/'confirmation/paired_baseline/results.json','uniform');cg=indexed(ROOT/'confirmation/candidate/results.json','uniform')
# Non-SONIC records carry source+command, reconstruct the predeclared command index.
nonsonic=json.loads((ROOT/'beyondmimic_confirmation_capacity256/results.json').read_text());assert len(nonsonic)==880;nonsonic={(r['source'],round(r['command'],8)):r for r in nonsonic};assert len(nonsonic)==880
rows=[]
for k,c in cm_manifest.items():
 b=bm_manifest[k];assert b['seed']==c['seed'] and b['command']==c['command'];r={key:c[key] for key in ['task','source','seed','command','command_index']};r.update(baseline_human=native_metrics(b),human=native_metrics(c),baseline_direction_g1=bd[k].get('g1'),baseline_direction=bd[k].get('actual'),baseline_gmr_g1=bg[k].get('g1'),baseline_gmr=bg[k].get('actual'),g1=cg[k].get('g1'),sonic=cg[k].get('actual'));alt=nonsonic[(c['source'],round(c['command'],8))];r['beyondmimic']=alt.get('actual');r['selected_method']=protocol['task_route'][r['task']];r['selected']=r[r['selected_method']];r['errors']={key:value for key,value in [('baseline_direction',bd[k].get('error')),('baseline_gmr',bg[k].get('error')),('sonic',cg[k].get('error')),('beyondmimic',alt.get('error'))] if value};rows.append(r)
keys=['baseline_human','human','baseline_direction_g1','baseline_direction','baseline_gmr_g1','baseline_gmr','g1','sonic','beyondmimic','selected'];summary={};rng=np.random.default_rng(92026)
for i,task in enumerate(TASKS):
 a=[r for r in rows if r['task']==task];assert len(a)==80;span=RANGES[i][1]-RANGES[i][0];summary[task]={key:stats(a,key,span,TOLS[i]) for key in keys}
 for key in keys:
  summary[task][key]['semantic_E_all']=float(np.mean([min(abs(r[key]['quantity']-r['command'])/span,1) if r[key] is not None and r[key]['event_pass'] else 1 for r in a]))
 for key in keys:
  complete=[];errors=[]
  for source in sorted({r['source'] for r in a}):
   s=sorted([r for r in a if r['source']==source],key=lambda r:r['command']);assert len(s)==5
   errors.append(np.mean([min(abs(r[key]['quantity']-r['command'])/span,1) if r[key] is not None else 1 for r in s]))
   if all(r[key] is not None for r in s):complete.append(all(s[j+1][key]['quantity']>=s[j][key]['quantity']-1e-6 for j in range(4)))
  summary[task][key]['monotonic_complete_sources']=sum(complete);summary[task][key]['sources_tested_for_monotonicity']=len(complete)
 # Paired source bootstrap; prompts shared, intervals describe prompt/noise set, not unseen-text claims.
 sources=sorted({r['source'] for r in a});deltas=[]
 for source in sources:
  s=[r for r in a if r['source']==source];er=lambda key:np.mean([min(abs(r[key]['quantity']-r['command'])/span,1) if r[key] is not None else 1 for r in s]);deltas.append(er('selected')-er('baseline_direction'))
 boot=np.mean(np.asarray(deltas)[rng.integers(0,16,(5000,16))],axis=1);summary[task]['paired_delta_E_all_selected_minus_baseline']=dict(mean=float(np.mean(deltas)),source_bootstrap_95=np.quantile(boot,[.025,.975]).tolist(),unit='prompt/noise pair, five commands kept together')
flat=[]
for r in rows:
 line={key:r[key] for key in ['task','source','seed','command','command_index','selected_method']}
 for key in keys:
  line[key+'_Q']=r[key]['quantity'] if r[key] else None;line[key+'_event']=r[key]['event_pass'] if r[key] else None
  if r[key]:
   for metric,value in r[key].items():
    if metric not in ['quantity','event_pass'] and isinstance(value,(int,float,bool,str)):line[key+'_'+metric]=value
 flat.append(line)
with (out/'all_requests.csv').open('w') as f:w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for line in flat for k in line)));w.writeheader();w.writerows(flat)
(out/'audited_results.json').write_text(json.dumps(rows,indent=2,default=lambda x:x.item()));(out/'summary.json').write_text(json.dumps(summary,indent=2,default=lambda x:x.item()));print('Assessed 880 paired requests; no failures excluded')
