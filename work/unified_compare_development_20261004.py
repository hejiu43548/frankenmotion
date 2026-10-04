"""Compare completed, audited models only on the same 110 development identities."""
from pathlib import Path
import json,argparse
import numpy as np
R=Path('/home/pku/frankenmotion');U=R/'outputs_amass/franken_unified_20261004';read=lambda p:json.loads(p.read_text())
p=argparse.ArgumentParser();p.add_argument('--names',nargs='+',default=['official_dance_validation','bm_native_routed_validation','joint_v1_validation','joint_v2_plain_validation','joint_v2_preview_validation','joint_v3_pose_common_validation']);a=p.parse_args();base=U/'evaluation/routed_baseline_validation';br=read(base/'audited_results.json');key=lambda r:(r['source'],r['command_index']);bi={key(r):r for r in br};assert len(bi)==110;results={'routed_baseline':read(base/'audit.json')['aggregate']};details={};pending=[]
for name in a.names:
 path=U/'evaluation'/name
 if not (path/'audit.json').exists():pending.append(name);continue
 audit=read(path/'audit.json');assert (audit['unique_checkpoints']==1 or audit.get('scope')=='BM native routed control') and audit['raw_max_error']<1e-8 and audit['overflow_warnings']==0;rows=read(path/'audited_results.json');ci={key(r):r for r in rows};assert len(ci)==110 and ci.keys()==bi.keys()
 for k,r in ci.items():
  b=bi[k];assert all(r[f]==b[f] for f in ['path','seed','task','command'])
  for f in ['human','g1']:assert abs(r[f]['quantity']-b[f]['quantity'])<1e-8 and r[f]['event_pass']==b[f]['event_pass']
 results[name]=audit['aggregate'];s=read(path/'summary.json');bs=read(base/'summary.json');details[name]={task:dict(complete=v['actual']['measurable'],events=v['actual']['event_pass'],joint=v['actual']['joint_pass'],semantic_E_all=v['actual']['semantic_E_all'],baseline_semantic_E_all=bs[task]['actual']['semantic_E_all'],slope=v['actual']['median_source_slope']) for task,v in s.items()}
roles={name:('routed diagnostic control, not eligible for unified selection' if name in ['routed_baseline','bm_native_routed_validation'] else ('initial single-policy control' if name=='official_dance_validation' else 'trained unified candidate')) for name in results}
report=dict(roles=roles,scope='Development-only model comparison; not final holdout',aggregate=results,per_task=details,pending=pending);(U/'development_comparison.json').write_text(json.dumps(report,indent=2));print(json.dumps(report,indent=2))
