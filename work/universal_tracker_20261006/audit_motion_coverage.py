"""Kinematic coverage diagnostic, not a feasibility or semantic classifier."""
import json
from pathlib import Path
import numpy as np
D=Path('/home/pku/frankenmotion/outputs_amass/universal_tracker_20261006')
def features(row,strip):
 with np.load(row['motion_path']) as z:
  q=z['joint_pos'][strip:];v=z['joint_vel'][strip:];pos=z['body_pos_w'][strip:,0];omega=z['body_ang_vel_w'][strip:,0];velocity=z['body_lin_vel_w'][strip:,0]
 return dict(pelvis_height_min=float(pos[:,2].min()),pelvis_height_max=float(pos[:,2].max()),root_speed_p95=float(np.quantile(np.linalg.norm(velocity[:,:2],axis=1),.95)),root_angular_speed_p95=float(np.quantile(np.linalg.norm(omega,axis=1),.95)),joint_velocity_p95=float(np.quantile(np.abs(v),.95)),joint_excursion_max=float(np.ptp(q,axis=0).max()))
groups={}
for tag,path in [('original_training','joint_corpus_v1/manifest.json'),('expanded_training','joint_corpus_expanded_corrected/manifest.json'),('natural_validation','natural_extended_corrected_motion/manifest.json')]:
 rows=json.loads((D/path).read_text());rows=[r for r in rows if tag!='natural_validation' or r['split']=='val'];records=[]
 for r in rows:
  records.append(dict(task=r['task'],source=r.get('source'),motion_path=r['motion_path'],features=features(r,50 if tag=='natural_validation' else 0)))
 groups[tag]=records
keys=list(groups['original_training'][0]['features']);envelopes={}
for tag,rows in groups.items():
 envelopes[tag]={k:dict(zip(['min','q05','median','q95','max'],map(float,np.quantile([r['features'][k] for r in rows],[0,.05,.5,.95,1])))) for k in keys}
out=dict(scope=__doc__,warning='Quantiles summarize per-clip extrema/percentiles, not frames; original training includes transitions. Distribution support is not proof of dynamic feasibility or generator semantics. Natural validation strips its artificial standing entry.',groups=groups,envelopes=envelopes)
(D/'motion_coverage_audit.json').write_text(json.dumps(out,indent=2));print(json.dumps(envelopes,indent=2))
