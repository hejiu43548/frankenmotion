"""Verify frozen weights, request pairing and realized checkpoint routes."""
from pathlib import Path
import json,hashlib
ROOT=Path('/home/pku/frankenmotion/outputs_amass/franken_improve_20261003')
def read(p):return json.loads(Path(p).read_text())
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
g=read(ROOT/'frozen_generation_v1/protocol.json');c=read(ROOT/'frozen_controllers_v1/protocol.json');wm=read(ROOT/'frozen_controllers_v1/weight_map.json')
for r in [*g['generators'].values(),*c['weights'].values()]:assert sha(r['frozen'])==r['sha256'],r['frozen']
b=read(ROOT/'paired_baseline_confirmation_manifest.json');a=read(ROOT/'candidate_confirmation_manifest.json');assert len(a)==len(b)==880
index=lambda rows:{(r['source'],r['command_index']):r for r in rows}
ai=index(a);bi=index(b);assert len(ai)==len(bi)==880 and ai.keys()==bi.keys()
for k,r in ai.items():
 old=bi[k];assert all(old[x]==r[x] for x in ['task','seed','command','command_index'])
 pi=int(r['source'].split('_p')[-1].split('_')[0]);si=int(r['source'].split('_s')[-1]);assert r['seed']==42026000+pi*100+si
 p=read(Path(r['path']).parent/'provenance.json');w=g['generators']['jump_aligned' if r['task']=='jump' else 'physical'];assert p['sha256']==w['sha256'] and p['weights']==w['frozen']
for folder,manifest,n in [('confirmation/paired_baseline',bi,1760),('confirmation/candidate',ai,880)]:
 rows=read(ROOT/folder/'results.json');assert len(rows)==n
 identities={(r['source'],r['command_index'],r['method']) for r in rows};assert len(identities)==n
 for r in rows:
  expected=manifest[(r['source'],r['command_index'])];assert all(r[x]==expected[x] for x in ['path','seed','command','task'])
 p=read(ROOT/folder/'protocol.json');assert p['controller']=='SONIC v1.1 mode0';assert p['manifest_sha256']==sha(p['manifest'])
rows=read(ROOT/'beyondmimic_confirmation/results.json');assert len(rows)==880 and len(index(rows))==880
for r in rows:
 expected=ai[(r['source'],r['command_index'])];assert r['input_path']==expected['path'] and r['seed']==expected['seed'] and r['command']==expected['command']
 if not r.get('error'):assert r['checkpoint']==wm[r['task']],r
record=dict(status='passed',paired_requests=880,physical_rollouts=3520,frozen_hashes_verified=len(g['generators'])+len(c['weights']),generation_routes_verified=880,beyondmimic_checkpoint_routes_verified=sum(not r.get('error') for r in rows),error_requests=sum(bool(r.get('error')) for r in rows),note='Errors retained; provenance/identities verified independently of performance. Frozen protocols retain historical descriptions from their creation times.')
(ROOT/'final_delivery').mkdir(exist_ok=True)
(ROOT/'final_delivery/protocol_audit.json').write_text(json.dumps(record,indent=2));print(json.dumps(record))
