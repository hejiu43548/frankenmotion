import json,hashlib
from pathlib import Path
import torch,numpy as np
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/gait_demo_20261005';f=D/'frozen';protocol=json.loads((f/'protocol.json').read_text());rows=json.loads((D/'final_random32/manifest.json').read_text());actor=torch.jit.load(str(f/'actor.pt')).eval();torch.set_num_threads(2);results={};parity=[]
for backend in ['selected','selected_cpu']:
 rr=[]
 for row in rows:
  run=Path(row['source'])/backend;r=json.loads((run/'result.json').read_text());audit=json.loads((run/'audit.json').read_text());gait=json.loads((run/'gait_metrics.json').read_text());assert r['checkpoint_sha256']==protocol['weights']['policy.pt'];assert audit['success']==r['success'];z=np.load(run/'actual.npz')
  with torch.inference_mode():pred=actor(torch.from_numpy(z['observations']).float()).numpy()
  err=float(abs(pred-z['actions']).max());assert err<2e-4,(run,err);parity.append(dict(run=str(run),frames=len(pred),action_max_error=err));rr.append(dict(result=r,audit=audit,gait=gait))
 results[backend]=dict(count=len(rr),successes=sum(x['result']['success'] for x in rr),complete=sum(x['result']['physical_complete'] for x in rr),mean_root_error_m=float(np.mean([x['result']['goal_error_m'] for x in rr])),mean_palm_error_m=float(np.mean([x['result']['palm_target_error_m'] for x in rr])),min_contact_s=min(x['result']['longest_hand_contact_s'] for x in rr),mean_gait={key:float(np.mean([x['gait'][key] for x in rr])) for key in ['knee_std_deg','knee_amplitude_ratio','clearance_p95_m','double_support_fraction','flight_fraction','mean_contact_point_slip_m_s']},cases=rr)
(D/'final_results.json').write_text(json.dumps(results,indent=2));(f/'parity_audit.json').write_text(json.dumps(dict(all64rollouts_match_same_exported_actor=True,max_error=max(r['action_max_error'] for r in parity),cases=parity),indent=2));print(json.dumps({k:{a:b for a,b in v.items() if a!='cases'} for k,v in results.items()},indent=2))
