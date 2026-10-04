"""Freeze one development-selected trained actor before creating final inputs."""
import argparse,datetime,hashlib,json,shutil
from pathlib import Path
R=Path('/home/pku/frankenmotion');U=R/'outputs_amass/franken_unified_20261004'
read=lambda p:json.loads(p.read_text());sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
p=argparse.ArgumentParser();p.add_argument('--candidates',nargs='+',required=True);a=p.parse_args()
assert not (U/'final_test').exists(),'Final input cohort already exists; cannot reselect on this cohort'
out=U/'frozen_unified';assert not out.exists();candidates=[]
for name in a.candidates:
    folder=U/'evaluation'/name;proto=read(folder/'protocol.json');audit=read(folder/'audit.json');summary=read(folder/'summary.json')
    assert name.startswith('joint_') and proto['split']=='development_validation' and not proto['task_routing']
    assert audit['unique_checkpoints']==1 and audit['raw_max_error']<1e-8 and audit['overflow_warnings']==0
    assert audit['aggregate']['requests']==110 and len(summary)==11
    checkpoint=Path(proto['checkpoint']);assert sha(checkpoint)==proto['sha256']
    assert (checkpoint.parent/'complete.json').exists() and (checkpoint.parent/'protocol.json').exists()
    candidates.append(dict(name=name,checkpoint=str(checkpoint),sha256=proto['sha256'],preview=proto['preview'],preview_offsets=proto.get('preview_offsets',[5,10,20] if proto['preview'] else []),aggregate=audit['aggregate'],audit_sha256=sha(folder/'audit.json'),evaluation_protocol_sha256=sha(folder/'protocol.json')))
chosen=min(candidates,key=lambda c:(c['aggregate']['macro_semantic_E_all'],c['name']))
out.mkdir();weight=out/'policy.pt';shutil.copy2(chosen['checkpoint'],weight);assert sha(weight)==chosen['sha256']
protocol=dict(frozen_at_utc=datetime.datetime.now(datetime.timezone.utc).isoformat(),checkpoint=str(weight),checkpoint_sha256=sha(weight),preview=chosen['preview'],preview_offsets=chosen['preview_offsets'],selected=chosen['name'],selection_rule='Lowest all-11-task macro semantic E_all on the 110 development requests; lexicographic name tie-break only. No per-task checkpoint selection.',candidates=candidates,development_manifest_sha256=sha(U/'development_validation/manifest.json'),study_protocol_sha256=sha(U/'protocol.json'),final_seed_base=94041000,final_requests=880,task_routing=False)
(out/'protocol.json').write_text(json.dumps(protocol,indent=2));print(json.dumps(protocol,indent=2))
