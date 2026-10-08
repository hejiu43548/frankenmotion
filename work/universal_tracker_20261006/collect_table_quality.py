"""Quality diagnostics on the frozen paired table test, without dropping failed requests."""
import json,sys,subprocess
from pathlib import Path
import numpy as np
R=Path('/home/pku/frankenmotion');D=R/'outputs_amass/universal_tracker_20261006';W=R/'work/universal_tracker_20261006';result={}
for tag in ['candidate','broad','stable']:
 folder=D/'table_evaluation'/('fresh_'+tag);s=json.loads((folder/'summary.json').read_text());slip=folder/'contact_slip_walk.json'
 if not slip.exists():
  with (folder/'contact_slip_walk.log').open('w') as log:subprocess.run([sys.executable,str(W/'audit_contact_slip.py'),'--run',str(folder),'--table','--segment','walk'],stdout=log,stderr=subprocess.STDOUT,check=True)
 contact=json.loads(slip.read_text());rows=[]
 for r in s['results']:
  if 'run' not in r:rows.append(dict(runtime_error=r));continue
  run=Path(r['run']);row=dict(index=r['scene']['index'],physical_complete=r['physical_complete'],full_flow_success=r['success'])
  for segment in ['walk','away']:
   p=run/(segment+'_jitter_metrics.json');j=json.loads(p.read_text()) if p.exists() else None;row[segment+'_torso_angular_velocity_highpass5hz_rms']=j['torso_angular_velocity']['highpass_5hz_rms'] if j else None
  match=next(x for x in contact['rows'] if x['source']=='scene_'+str(row['index']));row.update(walk_foot_contact_tangent_mean_m_s=match['mean_contact_tangent_speed_m_s'],walk_foot_contact_tangent_p95_m_s=match['p95_contact_tangent_speed_m_s']);rows.append(row)
 completed=[r for r in rows if r.get('physical_complete')];keys=['walk_torso_angular_velocity_highpass5hz_rms','away_torso_angular_velocity_highpass5hz_rms','walk_foot_contact_tangent_mean_m_s','walk_foot_contact_tangent_p95_m_s'];means={key:float(np.mean([r[key] for r in completed if r.get(key) is not None])) if any(r.get(key) is not None for r in completed) else None for key in keys};result[tag]=dict(planned=s['planned'],processed=s['processed'],complete=s['complete'],success=s['success'],completed_only_means=means,all_results=rows)
paired={}
for baseline in ['broad','stable']:
 left={r['index']:r for r in result['candidate']['all_results'] if r.get('physical_complete')};right={r['index']:r for r in result[baseline]['all_results'] if r.get('physical_complete')};ids=sorted(set(left)&set(right));paired[baseline]=dict(common_completed_indices=ids,requests=len(ids),candidate={k:float(np.mean([left[i][k] for i in ids])) for k in keys},baseline={k:float(np.mean([right[i][k] for i in ids])) for k in keys},scope='Same jointly completed scene subset; survival-conditioned quality comparison, not an all-request outcome.')
(D/'fresh_table_quality.json').write_text(json.dumps(dict(scope=__doc__,caution='Aggregate quality means are conditional on physical completion, with denominators and every failed request retained. Foot metric is geometrical contact-point tangent speed, not force-weighted slip or hardware validation.',controllers=result,paired_common_completed=paired),indent=2));print('Frozen table quality collected',flush=True)
